# Training Script - SLURM
It is recommended to use `screen` to make the session detachable, allowing you to exit and return to the interactive job(s).

## SBATCH Header
Slurm jobs are submitted via shell scripts that have a header specifying the resources the job needs. Here is an example header:

```
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --array=0-3
#SBATCH --time=04:15:00
#SBATCH --job-name=frcnn
#SBATCH --output=frcnn_%A_%a.out
```
See [https://slurm.schedmd.com/sbatch.html](https://slurm.schedmd.com/sbatch.html) for more information about using sbatch. 

## Activate Environment
After the header is where you place your code which will run on the resources the job scheduler assigns the job. 

Activating the proper virtual environment where the dependencies are installed

```
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate pytorch_env
```

**Note**: Make sure that the [Cuda](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) versions are compatible with the installed [Pytroch](https://pytorch.org/) version. 

## Run tasks
```
configs=("frcnn_slope.yml" "frcnn_slrm.yml" "frcnn_msrm_sf1.yml" "frcnn_msrm_sf2.yml")
config="${configs[$SLURM_ARRAY_TASK_ID]}"

python train.py --config="$config"
```


# Useful commands
```
srun --gres=gpu:0 --partition=gpuA100 --pty bash
```