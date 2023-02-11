#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=pytorch_faster_rcnn_setup
#SBATCH --output=faster_rcnn_setup.out
 
# Set up environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda create -n pytorch_env -c pytorch pytorch torchvision numpy -y

# generate yaml file
# conda activate <name_of_environment>
# conda env export > environment.yml

#create enviroment from yaml file
# conda env create -f environment.yml
# conda info --envs

## setup from requirements.txt

# 1. conda create --name <env_name> python=<version>
# 2. pip install requirements.txt
# 3. conda info -e
# 4. conda activate <environment_name>

# conda create --name <new_name> --clone <old_name>
# conda remove --name <old_name> --all

# conda env update --prune