from data_objects.preprocess import preprocess_voxceleb1
from data_objects.compute_mean_std import compute_mean_std
from data_objects.partition_voxceleb import partition_voxceleb
from pathlib import Path
import argparse
import subprocess
if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description="Preprocesses audio files from datasets.",
        formatter_class=MyFormatter
    )
    parser.add_argument("dataset_root", type=Path, help= \
        "Path to the directory containing VoxCeleb datasets. It should be arranged as:")
    parser.add_argument("-s", "--skip_existing", action="store_true", help= \
        "Whether to skip existing output files with the same name. Useful if this script was "
        "interrupted.")
    args = parser.parse_args()

    # Process the arguments

    #checking if all the folder and meta files are in place
    assert args.dataset_root.exists()
    assert args.dataset_root.joinpath('iden_split.txt').exists()
    assert args.dataset_root.joinpath('veri_test.txt').exists()
    #assert args.dataset_root.joinpath('vox1_meta.csv').exists()            #vox1_meta.csv is not required as there are only 2 classes(Male and Female)

    #joining paths so that required folders can be created at later stage.
    dev_out_dir = args.dataset_root.joinpath("feature", "dev")                   #Path('Audio_multilingual/feature/dev')
    test_out_dir = args.dataset_root.joinpath("feature", "test")                    #Path('Audio_multilingual/feature/test')
    merged_out_dir = args.dataset_root.joinpath("feature", "merged")                    #Path('Audio_multilingual/feature/merged')

    #making directories                                                                 features -> (dev, test, merged)
    dev_out_dir.mkdir(exist_ok=True, parents=True)          
    test_out_dir.mkdir(exist_ok=True, parents=True)
    merged_out_dir.mkdir(exist_ok=True, parents=True)       

    # Preprocess the datasets
    preprocess_voxceleb1(args.dataset_root, 'dev', dev_out_dir, args.skip_existing)
    preprocess_voxceleb1(args.dataset_root, 'test', test_out_dir, args.skip_existing)
    for path in dev_out_dir.iterdir():
        subprocess.call(['cp', '-r', path.as_posix(), merged_out_dir.as_posix()])
    for path in test_out_dir.iterdir():
        subprocess.call(['cp', '-r', path.as_posix(), merged_out_dir.as_posix()])
    compute_mean_std(merged_out_dir, args.dataset_root.joinpath('mean.npy'),
                     args.dataset_root.joinpath('std.npy'))
    partition_voxceleb(merged_out_dir, args.dataset_root.joinpath('iden_split.txt'))
    print("Done")
