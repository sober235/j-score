#!/bin/bash

conda_env="jscore"
conda create --name $conda_env python=3.10 -y
conda activate $conda_env

conda install cuda -c nvidia/label/cuda-11.6.2 -y
conda install cudatoolkit=11.3.1=h2bc3f7f_2 -y
conda install cudnn=8.2.1=cuda11.3_0 -y
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y

pip install scipy==1.9.3
pip install absl-py==1.3.0
pip install tensorboard
pip install tensorflow
pip install ml-collections
pip install mat73
pip install scikit-image
conda install h5py -y

conda deactivate
