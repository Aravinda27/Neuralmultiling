from distutils import extension
from multiprocess.pool import ThreadPool
from data_objects.params_data import *  
from datetime import datetime
from data_objects import audio
from pathlib import Path
# from tqdm import tqdm
import numpy as np

anglophone_nationalites = ["india"]

class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """
#-------"audio_multilingual/feature/dev" / "audio_multilingual/feature/test"----------"audio_multilingual"
    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" % name.replace("/", "_")), "w")
        self.sample_data = dict()

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()

    def _log_params(self):
        from data_objects import params_data
        self.write_line("Parameter values:")

        #Getting all parameter's value-----------dir() function returns all methods and properties name associated.
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):              
            value = getattr(params_data, param_name)                                        #getting the parameter value (say param_data is an object of a class and param_name is a variable in that class). There's a 3rd argument(optional) which is the default value if the param_name is not found
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)

    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()

#------------"audio_multilingual"----------path('audio_multilingual')--------Path('audio_multilingual/feature/dev') / Path('audio_multilingual/feature/test')-----
def _init_preprocess_dataset(dataset_name, dataset_root, out_dir):
    if not dataset_root.exists():           
        print("Couldn\'t find %s, skipping this dataset." % dataset_root)       #path to dataset doesn't exists
        return None, None
    return tuple([dataset_root, DatasetLog(out_dir, dataset_name)])


# -------speaker_dirs: [Path('audio_multilingual/dev/wav/2524M'),..]------dataset_name: "audio_multilingual"-------
# ------datasets_root:Path('audio_multilingual')-------out_dir:Path('audio_multilingual/feature/dev') / Path('audio_multilingual/feature/test')-----
# -------extension: "wav"-------logger: DatasetLog(out_dir, dataset_name)

def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension, skip_existing, logger):
    print("%s: Preprocessing data having %d audio files." % (dataset_name, len(speaker_dirs) * 6))

    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_dir: Path):          #Like, speaker_dir: Path('audio_multilingual/dev/wav/2524M')
        # Give a name to the speaker that includes its dataset
        speaker_name = speaker_dir.parts[-1]                                #speaker_name will be '2524M'

        # Create an output directory with that name, as well as a txt file containing a
        # reference to each source file.
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")

        # There's a possibility that the preprocessing was interrupted earlier, check if
        # there already is a sources file.
        if sources_fpath.exists():
            try:
                with sources_fpath.open("r") as sources_file:
                    existing_fnames = {line.split(",")[0] for line in sources_file}
            except:
                existing_fnames = {}
        else:
            existing_fnames = {}
            
        # Gather all audio files for that speaker recursively
        sources_file = sources_fpath.open("a" if skip_existing else "w")
        for in_fpath in speaker_dir.glob("**/*.%s" % extension):               #in_fpath: Path('audio_multilingual/dev/wav/2524M/iphone10/2524M_iphone10_session2_bengali_1.wav')
            # Check if the target output file already exists
            #out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)   #speaker_dir: Path('audio_multilingual/dev/wav/2524M'), so in_fpath.relative_to(speaker_dir) will give Path('iphone10/2524M_iphone10_session2_bengali_1.wav')
            out_fname = in_fpath.parts[-1]      #name of the audio file contains all information.
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue

            # Load and preprocess the waveform
            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                print(in_fpath)
                continue

            # Create the mel spectrogram, discard those that are too short
            # frames = audio.wav_to_mel_spectrogram(wav)
            frames = audio.wav_to_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue

            out_fpath = speaker_out_dir.joinpath(out_fname)     #out_fname: 2524M_iphone10_session2_bengali_1.wav
            np.save(out_fpath, frames)
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))

        sources_file.close()

    # Process the utterances for each speaker
    preprocess_speaker(speaker_dirs[0])
    #for audio_path in speaker_dirs:
       # preprocess_speaker(audio_path)
    with ThreadPool(1) as pool:
        list(pool.imap(preprocess_speaker, speaker_dirs))
    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)


#--------Path('audio_multilingual')------ "dev"/"test"---Path('audio_multilingual/feature/dev') / Path('audio_multilingual/feature/test')-----
def preprocess_voxceleb1(dataset_root: Path, parition: str, out_dir: Path, skip_existing=False):
    # Initialize the preprocessing
    dataset_name = "audio_multilingual"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, dataset_root, out_dir)
    if not dataset_root:
        return

    #----------- No need because we don't have to choose some country specific speakers only

    # Get the contents of the meta file
    #with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
    #    metadata = [line.split("\t") for line in metafile][1:]
    

    # Select the ID and the nationality, filter out non-anglophone speakers
    #nationalities = {line[0]: line[3] for line in metadata}
    #keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if nationality.lower() in anglophone_nationalites]                #keeping only those speakers who are from above mentioned countries
                        
    #----------------------------------------------------------------------------------

    #print(f"{dataset_name}: using samples from %d (presumed anglophone) speakers out of %d." % (len(keep_speaker_ids), len(nationalities)))

    # Get the speaker directories for anglophone speakers only
    speaker_dirs = dataset_root.joinpath(parition, 'wav').glob("*")             #(* i.e., All) directories and files in the folder (It doesn't list out the directories and files inside the directory of that folder). In our case there will be only folders(inside VoxCeleb1/dev/wav)
    
    #speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if speaker_dir.parts[-1] in keep_speaker_ids]
    speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs]

    print(f"{dataset_name}: found {len(speaker_dirs) * 6} audio files on the disk.")        # As there are 6 audio file in each last innermost folder.
    # Preprocess all speakers
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, dataset_root, out_dir, "wav", skip_existing, logger)

