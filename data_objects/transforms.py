import numpy as np
import random

class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, input):
        return (input - self.mean) / self.std


class TimeReverse(object):
    def __init__(self, p=0.5):
        super(TimeReverse, self).__init__()
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return np.flip(input, axis=0).copy()
        return input


def generate_test_sequence(feature, partial_n_frames, shift=None):
    while feature.shape[0] <= partial_n_frames:				# Shape[0] of feature array is not fixed. It could be [665, 257], [429,257], ...    Say it is [665, 257] for this audio file
        feature = np.repeat(feature, 2, axis=0)				# Until no. of rows in features is less than partial_n_frame, append the feature matrix again i.e. first time it will be [1330, 257], Next time it will be [2660, 257]
        
    if shift is None:
        shift = partial_n_frames // 2
    test_sequence = []
    start = 0
    while start + partial_n_frames <= feature.shape[0]:			# The test_sequence will be of shape [600, 257]. It makes the number of rows mulitple of parital_n_frame // 2 
        test_sequence.append(feature[start: start + partial_n_frames])
        start += shift
    test_sequence = np.stack(test_sequence, axis=0)         #It stacks all those elements of test_sequence of [shape partial_n_frames//2, 257] into a single matrix of shape [partial_n_frame * k, 257]
    return test_sequence
