from data_objects.speaker import Speaker
import numpy as np

#dataset_dir: Path('Audio_multilingual/feature/merged'), ----- output_path_mean: Path('Audio_multilingual/mean.npy'), --- output_path_std: Path('Audio_multilingual/std.npy')
def compute_mean_std(dataset_dir, output_path_mean, output_path_std):
    print("Computing mean std...")
    speaker_dirs = [f for f in dataset_dir.glob("*") if f.is_dir()]             #speaker_dirs: [Path('Audio_multilingual/feature/merged/2524M'), .. ]
    if len(speaker_dirs) == 0:
        raise Exception("No speakers found. Make sure you are pointing to the directory "
                        "containing all preprocessed speaker directories.")
    speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]

    sources = []
    for speaker in speakers:
        sources.extend(speaker.sources)             #For a particular speaker (like Path('Audio_multilingual/feature/merged/2524M')) speaker.sources will be a list of : [['2524M_iphone10_session2_bengali_1.npy', 'audio_multilingual/dev/wav/2524M/iphone10/2524M_iphone10_session2_bengali_1.wav'], ..]
                                                    
                                                    #sources will be a list of all such speaker.sources: [['2524M_iphone10_session2_bengali_1.npy', 'audio_multilingual/dev/wav/2524M/iphone10/2524M_iphone10_session2_bengali_1.wav'], ..]
                                                    #sources will be for all speakers combined
    sumx = np.zeros(257, dtype=np.float32)
    sumx2 = np.zeros(257, dtype=np.float32)
    count = 0
    n = len(sources)
    for i, source in enumerate(sources):                        #source is an iterator of type containing element like: ['2524M_iphone10_session2_bengali_1.npy', 'audio_multilingual/dev/wav/2524M/iphone10/2524M_iphone10_session2_bengali_1.wav']
        feature = np.load(source[0].joinpath(source[1]))
        sumx += feature.sum(axis=0)
        sumx2 += (feature * feature).sum(axis=0)
        count += feature.shape[0]

    mean = sumx / count
    std = np.sqrt(sumx2 / count - mean * mean)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    np.save(output_path_mean, mean)
    np.save(output_path_std, std)
