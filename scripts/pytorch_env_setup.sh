#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=env_setup
#SBATCH --output=env_setup.out
 
# Set up environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda env create --file environment.yml
# conda create -n pytorch_env -c pytorch pytorch torchvision numpy -c conda-forge albumannotations matplotlib pandas opencv_python PyYAML -y
