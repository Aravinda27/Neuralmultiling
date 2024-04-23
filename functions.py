from cmath import isnan
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import matplotlib.pyplot as plt

from utils import compute_eer
from utils import AverageMeter, ProgressMeter, accuracy

plt.switch_backend('agg')
logger = logging.getLogger(__name__)


def train(cfg, model, optimizer, train_loader, val_loader, criterion, architect, epoch, writer_dict, lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')			#(Name, Format)
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    alpha_entropies = AverageMeter('Entropy', ':.4e')
    #progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1.val, top5.val, alpha_entropies, prefix="Epoch: [{}]".format(epoch), logger=logger)
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1.val, alpha_entropies, prefix="Epoch: [{}]".format(epoch), logger=logger)
    writer = writer_dict['writer']

    # switch to train mode
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        if lr_scheduler:
            current_lr = lr_scheduler.set_lr(optimizer, global_steps, epoch)
        else:
            current_lr = cfg.TRAIN.LR

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)          
        target = target.cuda(non_blocking=True)       

        input_search, target_search = next(iter(val_loader))
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        # step architecture
        architect.step(input_search, target_search)

        alpha_entropy = architect.model.module.compute_arch_entropy()
        alpha_entropies.update(alpha_entropy.mean(), input.size(0))

        # compute output
        output = model(input)

        # measure accuracy and record loss
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target)
        top1.update(acc1[0], input.size(0))
        #top5.update(acc5[0], input.size(0))
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to logger
        writer.add_scalar('lr', current_lr, global_steps)
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer.add_scalar('arch_entropy', alpha_entropies.val, global_steps)

        writer_dict['train_global_steps'] = global_steps + 1

        # log acc for cross entropy loss
        writer.add_scalar('train_acc1', torch.mean(top1.val).item(), global_steps)
        #writer.add_scalar('train_acc5', torch.mean(top5.val).item(), global_steps)

        if i % cfg.PRINT_FREQ == 0:
            progress.print(i)


def train_from_scratch(cfg, model, optimizer, train_loader, criterion, epoch, writer_dict, lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    #progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1.val, top5.val, prefix="Epoch: [{}]".format(epoch), logger=logger)
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1.val, prefix="Epoch: [{}]".format(epoch), logger=logger)
    writer = writer_dict['writer']

    # switch to train mode
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        assert not torch.isnan(input).any().item()
        global_steps = writer_dict['train_global_steps']

        if lr_scheduler:
            current_lr = lr_scheduler.get_last_lr()
        else:
            current_lr = cfg.TRAIN.LR

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output: Forward Propagation
        output = model(input)
        #print('training from scratch: ' + str(i))

        # measure accuracy and record loss
        loss = criterion(output, target)
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))  
        acc1 = accuracy(output, target)                     #As there are only 2 classes(M & F), top5 prediction doesn't make sense. Also we will be only having acc1 then.
        top1.update(acc1[0], input.size(0))
        #top5.update(acc5[0], input.size(0))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to logger
        writer.add_scalar('lr', current_lr, global_steps)
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        # log acc for cross entropy loss
        writer.add_scalar('train_acc1', torch.mean(top1.val).item(), global_steps)
        #writer.add_scalar('train_acc5', torch.mean(top5.val).item(), global_steps)

        if i % cfg.PRINT_FREQ == 0:
            progress.print(i)


def validate_verification(cfg, model, test_loader):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(len(test_loader), batch_time, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.eval()
    labels, distances = [], []

    with torch.no_grad():
        correct = 0
        threshold = 0.5
        end = time.time()
        for i, (input1, input2, label) in enumerate(test_loader):
            input1 = input1.cuda(non_blocking=True).squeeze(0)
            input2 = input2.cuda(non_blocking=True).squeeze(0)
            label = label.cuda(non_blocking=True)

            # compute output
            outputs1 = model(input1).mean(dim=0).unsqueeze(0)
            outputs2 = model(input2).mean(dim=0).unsqueeze(0)

            #print(outputs1.size())
            #print(outputs2)
            

            dists = F.cosine_similarity(outputs1, outputs2)
            #print(dists)
            
            #-------------------------To print how many times model predicted incorrectly-------------------------------
                #cosine similarity can give value in between [-1, 1]. labels is either 0(not same speaker) or 1(same speaker), so after rounding the tensor, passing it through the ReLU function so that -ve number(different speakers predicted) should get changed 0 and +ve numbers will remains as it is.
            
            #correct += torch.eq(F.relu(dists.round()), label).sum().item()

            if label.item():                                    #if label is 1, then it is same speaker
                correct += 1 if dists > threshold else 0

            else:                                               #if label is 0, then it is different speaker
                correct += 1 if dists <= threshold else 0

            #-----------------------------------------------------------------------------------------------------------
            dists = dists.data.cpu().numpy()
            distances.append(dists)
            labels.append(label.data.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 1000 == 0:
                progress.print(i)

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        fpr, tpr, fmr, fnmr, eer = compute_eer(distances, labels)
        
        logger.info('Test EER: {:.8f}'.format(np.mean(eer)))     
        logger.info(f'Incorrect Recognitions: {(len(labels) - correct)}')        #incorrects: len(labels) - correct

    return fpr, tpr, fmr, fnmr, eer


def validate_identification(cfg, model, test_loader, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    #progress = ProgressMeter(len(test_loader), batch_time, losses, top1.val, top5.val, prefix='Test: ', logger=logger)
    progress = ProgressMeter(len(test_loader), batch_time, losses, top1.val, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda(non_blocking=True).squeeze(0)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.mean(output, dim=0, keepdim=True)
            output = model.module.forward_classifier(output)
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = accuracy(output, target)
            top1.update(acc1[0], input.size(0))
            #top5.update(acc5[0], input.size(0))
            loss = criterion(output, target)

            losses.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)
        
        #logger.info('Test Acc@1: {:.8f} Acc@5: {:.8f}'.format(top1.avg[0].item(), top5.avg[0].item()))
        logger.info('Test Acc@1: {:.8f}'.format(top1.avg[0].item()))

    return top1.avg

