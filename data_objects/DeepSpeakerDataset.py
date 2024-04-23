from __future__ import print_function


import numpy as np
import torch.utils.data as data
from data_objects.speaker import Speaker
from torchvision import transforms as T
from data_objects.transforms import Normalize, TimeReverse, generate_test_sequence


def find_classes(speakers):
    #classes = list(set([speaker.name for speaker in speakers]))         
    classes = list(set([speaker.name for speaker in speakers]))                 # speaker.name: '2524M', 

    classes.sort()          #for audio_multingual dataset, classes: {'2524M', '0321F',...}              
    class_to_idx = {classes[i]: i for i in range(len(classes))}   #{VoxCelebID: position(i) in sorted order, ..}
    #-------------For audio_multilingual----------
    #class_to_idx : { '2524M' : 0
    #                 '0321F' : 1,
    #                           ,
    #                           ,
    #                            }           

    #*********************************************
    return classes, class_to_idx


class DeepSpeakerDataset(data.Dataset):
    # data_dir: Path('audio_multilingual'),---- sub_dir: 'dev'/'merged',--- partial_n_frames: 300
    def __init__(self, data_dir, sub_dir, partial_n_frames, partition=None, language = None, is_test=False, deviceID=None):
        super(DeepSpeakerDataset, self).__init__()
        self.data_dir = data_dir                                    # self.data_dir: Path('audio_multilingual')
        self.root = data_dir.joinpath('feature', sub_dir)               # self.root: Path('audio_multilingual/feature/dev')
        self.partition = partition
        self.partial_n_frames = partial_n_frames                                # self.partial_n_frames: 300
        self.is_test = is_test
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]               #speaker_dirs: [Path('audio_multilingual/feature/dev/0221M'), Path('audio_multilingual/feature/dev/10306F'), ... ]
       
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speakers = [Speaker(speaker_dir, self.partition, language, deviceID) for speaker_dir in speaker_dirs]

        classes, class_to_idx = find_classes(self.speakers)
        sources = []
        for speaker in self.speakers:
            sources.extend(speaker.sources)			#sources is a list of list
        self.features = []
        for source in sources:
            #item = (source[0].joinpath(source[1]), class_to_idx[source[2]])          #(Path of npy file, position of VoxCelebID folder in sorted order)
            item = (source[0].joinpath(source[1]), class_to_idx[source[2]])          # (Path('audio_multilingual/feature/dev/0221M/0221M_iphone6s_session2_hindi_4.npy'), class_to_idx['2524M'])
            self.features.append(item)
        mean = np.load(self.data_dir.joinpath('mean.npy'))
        std = np.load(self.data_dir.joinpath('std.npy'))
        self.transform = T.Compose([
            Normalize(mean, std),                         #Clubs mean and std in a single object
            TimeReverse(),                                      #only p = 0.5, one variable.
        ])						              #Transformations that needs to be applied.

    # Path('audio_multilingual/feature/dev/0221M/0221M_iphone6s_session2_hindi_4.npy') --- class_to_idx['M'](or 1)
    def load_feature(self, feature_path, speaker_id):
        feature = np.load(feature_path)             #feature's 0th dim is not fixed. It can be [657, 257], [429, 257], ..
        if self.is_test:
            test_sequence = generate_test_sequence(feature, self.partial_n_frames)      #Partial_n_frames is 300 as specified by the search.yaml
            return test_sequence, speaker_id
        else:
            if feature.shape[0] <= self.partial_n_frames:
                start = 0
                while feature.shape[0] < self.partial_n_frames:
                    feature = np.repeat(feature, 2, axis=0)
            else:
                start = np.random.randint(0, feature.shape[0] - self.partial_n_frames)          #pick a random between 0 and feature.shape[0] - 300
            end = start + self.partial_n_frames
            return feature[start:end], speaker_id           #Take the [start to end] part of feature i.e., feature shape will be [300, 257]

    def __getitem__(self, index):
        feature_path, speaker_id = self.features[index]                 # (Path('audio_multilingual/feature/dev/0221M/0221M_iphone6s_session2_hindi_4.npy'), class_to_idx['M'])
        feature, speaker_id = self.load_feature(feature_path, speaker_id)

        if self.transform is not None:
            feature = self.transform(feature)
        return feature, speaker_id                  #feature shape: [300, 257]

    def __len__(self):
        return len(self.features)

