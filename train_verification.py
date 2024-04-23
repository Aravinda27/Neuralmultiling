# -*- coding: utf-8 -*-
# @Date    : 2019-08-09
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import datetime
import shutil
import os
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.model import Network
from config import cfg, update_config
from utils import set_path, create_logger, save_checkpoint, count_parameters, Genotype
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset
from data_objects.VoxcelebTestset import VoxcelebTestset
from functions import train_from_scratch, validate_verification
from loss import CrossEntropyLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train energy network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--lang',
                        help="Language of the audios to be used during verification, (lang -- lang) would be verified",
                        default=None,
                        type=str)           #default is None which means all languages (lang1 -- lang2) would be also verified

    parser.add_argument('--load_path',
                        help="The path to resumed dir",
                        default=None)
    
    parser.add_argument('--device',
                        help="all audio should be from the given device",
                        default=None,
                        type=str)               #default is none which means across all device

    parser.add_argument('--text_arch',                  #Type is String
                        help="The text to arch",
                        default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    assert args.text_arch               #Architecture needs to be provided

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Set the random seed manually for reproducibility.
    current_time = datetime.datetime.now()
    np.random.seed(current_time.microsecond)
    torch.manual_seed(current_time.microsecond)
    torch.cuda.manual_seed_all(current_time.microsecond)

    # Loss
    criterion = CrossEntropyLoss(cfg.MODEL.NUM_CLASSES).cuda()

    # load arch
    genotype = eval(args.text_arch)         #Getting the architecture of the genotype

    model = Network(cfg.MODEL.INIT_CHANNELS, cfg.MODEL.NUM_CLASSES, cfg.MODEL.LAYERS, genotype)
    model = model.cuda()

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.LR
    )

    # resume && make log dir and logger
    if args.load_path and os.path.exists(args.load_path):
        #checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint_best.pth')

        #To start from last checkpoint----------------------------------
        checkpoints = os.listdir(os.path.join(args.load_path, 'Model'))
        assert len(checkpoints) != 0
        checkpoint = max([int(item.split('_')[1].split('.')[0]) for item in checkpoints if item != 'checkpoint_best.pth'])
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint_' + str(checkpoint) + '.pth')

        #---------------------------------------------------------------

        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        # load checkpoint
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_eer = checkpoint['best_eer']
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.path_helper = checkpoint['path_helper']

        logger = create_logger(args.path_helper['log_path'])
        logger.info("=> loaded checkloggpoint '{}'".format(checkpoint_file))
    else:
        exp_name = args.cfg.split('/')[-1].split('.')[0]
        args.path_helper = set_path('logs_scratch', exp_name)
        logger = create_logger(args.path_helper['log_path'])
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        best_eer = 1.0
        last_epoch = -1
    logger.info(args)
    logger.info(cfg)
    logger.info(f"selected architecture: {genotype}")
    logger.info("Number of parameters: {}".format(count_parameters(model))) 

    # dataloader
    train_dataset = DeepSpeakerDataset(
        Path(cfg.DATASET.DATA_DIR),  cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES, language=args.lang, deviceID=args.device)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    test_dataset_verification = VoxcelebTestset(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.PARTIAL_N_FRAMES, cfg.DATASET.TEST_FILE)
    test_loader_verification = torch.utils.data.DataLoader(
        dataset=test_dataset_verification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # training setting
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // cfg.VAL_FREQ,
    }

    # training loop
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, cfg.TRAIN.LR_MIN,
        last_epoch=last_epoch
    )

    for epoch in tqdm(range(begin_epoch, cfg.TRAIN.END_EPOCH), desc='train progress'):
        model.train()
        model.drop_path_prob = cfg.MODEL.DROP_PATH_PROB * epoch / cfg.TRAIN.END_EPOCH
        #fpr, tpr, fmr, fnmr, eer = validate_verification(cfg, model, test_loader_verification)
        
        train_from_scratch(cfg, model, optimizer, train_loader, criterion, epoch, writer_dict)

        if epoch % cfg.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH - 1:
            fpr, tpr, fmr, fnmr, eer = validate_verification(cfg, model, test_loader_verification)

            # remember best acc@1 and save checkpoint
            is_best = eer < best_eer
            best_eer = min(eer, best_eer)

            # save
            logger.info('=> saving checkpoint to {}'.format(args.path_helper['ckpt_path']))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_eer': best_eer,
                'optimizer': optimizer.state_dict(),
                'path_helper': args.path_helper,
                'genotype': genotype
            }, is_best, args.path_helper['ckpt_path'], 'checkpoint_{}.pth'.format(epoch))

        #saving fpr, tpr, fmr, fnmr results in np array for future graph plotting
        np.save(os.path.join(args.path_helper['ckpt_path'], 'fpr_{}'.format(epoch)), fpr)
        np.save(os.path.join(args.path_helper['ckpt_path'], 'tpr_{}'.format(epoch)), tpr)
        np.save(os.path.join(args.path_helper['ckpt_path'], 'fmr_{}'.format(epoch)), fmr)
        np.save(os.path.join(args.path_helper['ckpt_path'], 'fnmr_{}'.format(epoch)), fnmr)

        if is_best:
            np.save(os.path.join(args.path_helper['ckpt_path'], 'Best_fpr'), fpr)
            np.save(os.path.join(args.path_helper['ckpt_path'], 'Best_tpr'), tpr)
            np.save(os.path.join(args.path_helper['ckpt_path'], 'Best_fmr'), fmr)
            np.save(os.path.join(args.path_helper['ckpt_path'], 'Best_fnmr'), fnmr)
            
        lr_scheduler.step(epoch)


if __name__ == '__main__':
    main()
