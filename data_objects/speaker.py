from data_objects.utterance import Utterance
from pathlib import Path


# Contains the set of utterances of a single speaker
class Speaker:
    # root: Path('Audio_multilingual/feature/merged/2524M'), -- partition: None
    def __init__(self, root: Path, partition=None, language = None, deviceID = None):
        self.root = root
        self.partition = partition
        self.name = root.name		#self.name: '2524M'
        self.utterances = None
        self.utterance_cycler = None
        sourcess = []
        if self.partition is None:
            #Audio_multilingual/feature/merged/2524M/_source.txt will be a file containing the entries for a unique identifier folder (like 2524M)
            #It is like: [2524M_iphone10_session2_bengali_1.npy, audio_multilingual/dev/wav/2524M/iphone10/2524M_iphone10_session2_bengali_1.wav]
            #            [                                       .....                                                                          ]
            #            [                                       .....                                                                          ]
            #            [                                       .....                                                                          ]

            with self.root.joinpath("_sources.txt").open("r") as sources_file:
                for l in sources_file:
                    tmp = l.strip().split(",")
                    if (language is None or tmp[-1].find(language) != -1) and (deviceID is None or tmp[-1].find(deviceID) != -1):
                        sourcess.append(tmp)
                #sources = [l.strip().split(",") for l in sources_file]              #[['2524M_iphone10_session2_bengali_1.npy', 'audio_multilingual/dev/wav/2524M/iphone10/2524M_iphone10_session2_bengali_1.wav'], ..]
        else:
            with self.root.joinpath("_sources_{}.txt".format(self.partition)).open("r") as sources_file:
                for l in sources_file:
                    tmp = l.strip().split(",")
                    if language is None or tmp[-1].find(language) != -1:
                        sourcess.append(tmp)
                #sources = [l.strip().split(",") for l in sources_file]                                            #something like this [[0_laIeN-Q44_00001.npy,VoxCeleb1/dev/wav/id10002/0_laIeN-Q44/00001.wav], ..]
        self.sources = [[self.root, frames_fname, self.name, wav_path] for frames_fname, wav_path in sourcess]        #[[Path to VoxCelebID folder containing npy files, .npy file, VoxCelebID, AudioFilePath], ..]
                        
                        #Path('Audio_multilingual/feature/merged/2524M'), '2524M_iphone10_session2_bengali_1.npy', '2524M', 'audio_multilingual/dev/wav/2524M/iphone10/2524M_iphone10_session2_bengali_1.wav'

    def _load_utterances(self):
        self.utterances = [Utterance(source[0].joinpath(source[1])) for source in self.sources]

    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all
        utterances come up at least once every two cycles and in a random order every time.

        :param count: The number of partial utterances to sample from the set of utterances from
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance,
        frames are the frames of the partial utterances and range is the range of the partial
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a
