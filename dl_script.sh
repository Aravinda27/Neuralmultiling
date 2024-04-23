#/bin/bash

# Contributed by Aaron Soellinger
# Usage (*nix):
# $ mkdir VoxCeleb1; cd VoxCeleb1; /bin/bash path/to/dl_script.sh "yourusername" "yourpassword"
# Note: I found my username and password in an email titled "VoxCeleb dataset" 

U="voxceleb1912"
P="0s42xuw6"
# wget http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/   --user $U --password $P &
# wget http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_dev_wav_partab --user $U --password $P &
# wget http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_dev_wav_partac --user $U --password $P &
# wget http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_dev_wav_partad --user $U --password $P &
# wget http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_test_wav.zip --user $U --password $P &
zip vox1_dev_wav.zip vox1_dev*
unzip vox1_dev_wav.zip -d "dev" &
unzip vox1_test_wav.zip -d "test"
# rm vox1_dev_wav_part*
rm wget*
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt
