#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=04:15:00
#SBATCH --job-name=frcnn
#SBATCH --output=frcnn.out

uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda3-py39
conda activate pytorch_env


python -u src/train.py --config=troubleshoot.yml