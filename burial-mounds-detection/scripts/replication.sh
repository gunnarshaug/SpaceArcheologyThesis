#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --array=0-3
#SBATCH --time=04:15:00
#SBATCH --job-name=frcnn
#SBATCH --nodelist=gorina8
#SBATCH --output=frcnn_%A_%a.out

uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda3-py39
conda activate pytorch_env

configs=("replication_slope.yml" "replication_slrm.yml" "replication_msrm_sf1.yml" "replication_msrm_sf2.yml")
config="${configs[$SLURM_ARRAY_TASK_ID]}"

python -u src/train.py --config="$config"