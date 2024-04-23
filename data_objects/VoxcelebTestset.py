import os
import torch.utils.data as data
import numpy as np
from torchvision import transforms as T
from data_objects.transforms import Normalize, generate_test_sequence

# pairs_path: Path('audio_multilingual/veri_test.txt'), db_dir: Path('audio_multilingual/feature/test')
def get_test_paths(pairs_path, db_dir):
    def convert_folder_name(path):                  #path will be like: '5229M/iphone11/5229M_iphone11_session2_hindi_4.wav'
        basename = os.path.splitext(path)[0]            #os.path.splitetxt is used to split the path name into pair (string of path(without extension of file), extension)  --- so basename will be ('5229M/iphone11/5229M_iphone11_session2_hindi_4', '.wav')
        items = basename.split('/')                         #items: ['5229M', 'iphone11', '5229M_iphone11_session2_hindi_4']
        speaker_dir = items[0]                              
        fname = items[-1] + '.npy'         #fname: '5229M_iphone11_session2_hindi_4.npy'
        p = os.path.join(speaker_dir, fname)                #p: '5229M/5229M_iphone11_session2_hindi_4.npy'
        return p

    pairs = [line.strip().split() for line in open(pairs_path, 'r').readlines()]            #pairs will be a list which will look like [['1', '5229M/iphone11/5229M_iphone11_session2_hindi_4.wav', '9204M/iphone10/9204M_iphone10_session1_hindi_5.wav'], ... ]
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    for pair in pairs:
        if pair[0] == '1':
            issame = True
        else:
            issame = False

        path0 = db_dir.joinpath(convert_folder_name(pair[1]))       # path0: Path('audio_multilingual/feature/test/5229M/5229M_iphone11_session2_hindi_4.npy')
        path1 = db_dir.joinpath(convert_folder_name(pair[2]))       # path1: Path('audio_multilingual/feature/test/9204M/9204M_iphone10_session1_hindi_5.npy')

        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list.append((path0,path1,issame))
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list


class VoxcelebTestset(data.Dataset):
    # data_dir: Path('audio_multilingual'), partial_n_frames: 300
    def __init__(self, data_dir, partial_n_frames, veri_test_file):
        super(VoxcelebTestset, self).__init__()
        self.data_dir = data_dir
        self.root = data_dir.joinpath('feature', 'test')                #self.root: Path('audio_multilingual/feature/test')
        self.test_pair_txt_fpath = data_dir.joinpath(veri_test_file)           #self.test_pair_txt_fpath: Path('audio_multilingual/veri_test.txt')
        self.test_pairs = get_test_paths(self.test_pair_txt_fpath, self.root)         #self.test_pairs will be a list of list which will look like [[Path('audio_multilingual/feature/test/5229M/5229M_iphone11_session2_hindi_4.npy'), Path('audio_multilingual/feature/test/9204M/9204M_iphone10_session1_hindi_5.npy'), 1], ... ]
        self.partial_n_frames = partial_n_frames
        mean = np.load(self.data_dir.joinpath('mean.npy'))
        std = np.load(self.data_dir.joinpath('std.npy'))
        self.transform = T.Compose([
            Normalize(mean, std)
        ])

    def load_feature(self, feature_path):
        feature = np.load(feature_path)
        test_sequence = generate_test_sequence(feature, self.partial_n_frames)
        return test_sequence

    def __getitem__(self, index):
        (path_1, path_2, issame) = self.test_pairs[index]

        feature1 = self.load_feature(path_1)
        feature2 = self.load_feature(path_2)

        if self.transform is not None:
            feature1 = self.transform(feature1)
            feature2 = self.transform(feature2)
        return feature1, feature2, issame

    def __len__(self):
        return len(self.test_pairs)
