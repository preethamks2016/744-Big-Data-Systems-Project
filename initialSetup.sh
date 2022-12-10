sudo apt update
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
exec bash
conda install numpy
conda install opencv
#if facing errors while running - ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y

#install torch 
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip3 install torch
pip install matplotlib
pip3 install transformers
#image dataset 
https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-0.tar.gz
tar -xvf train-0.tar.gz 
#follow this to install nvidia driver - https://piazza.com/class/l79l1xxtu6i1kb/post/100
#for nlp
pip3 install datasets
