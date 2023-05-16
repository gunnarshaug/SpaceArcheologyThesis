# Deep Learning with Airborne LiDAR data
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#install-dependencies">Install Dependencies</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#train">Train and Test</a></li>
  </ol>
</details>


## About The Project
The purpose of this study is to improve previous work conducted on detecting ancient burial mounds in Norway by utilizing a faster R-CNN object detection model with LiDAR-based Digital Terrain Models and identifying methods to reduce false positives.


## Prerequisites
The following lists the prerequisites to follow along this guide. Note that there exists various alternative tools to use. For example `QGIS` have similar functionalities like `ArcGIS Pro` and may be used instead. This guide will assume the following are installed:
- [Conda](https://docs.conda.io/en/latest/) - package management and environment management system. 
- [ArcGIS Pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/resources) - desktop GIS application from Esri.
- [Slurm](https://slurm.schedmd.com/) - an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for Linux clusters. 
  
## Install Dependencies

Install the dependencies with `conda` using a batch script to submit the job to the slurm-cluster.
```
sbatch scripts/setup.sh
```

Alternatively use the following command to create the conda environment:
```
conda env create --file environment.yml
```
**Note**: Make sure that the [Cuda](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) versions are matching with the `cudatoolkit` version and are compatible with the installed [Pytroch](https://pytorch.org/) version.

## Set up W&B for logging

[Weights & Biases](https://wandb.ai/) is a logging and visualization tool that helps users track and visualize machine learning experiments. It provides a centralized and standardized way to log and analyze experiment results, making it easy to compare different models, hyperparameters, and data sets. Wandb supports a variety of ML frameworks, including TensorFlow, PyTorch, and scikit-learn, and can be easily integrated into existing codebases. Wandb also provides a suite of features, including hyperparameter sweeps, anomaly detection, and dashboarding, to help users optimize and iterate on their models quickly and efficiently.

**Note:** The log files are saved both locally and remotely on the wandb server. The local files are saved in a folder called `wandb` by default and can safely be deleted if desired. I did not found a way to prevent it from logging locally. 

The following steps shows how to configure Weights & Biases:
1. Verify that wandb are installed within the environment:
  ```
  conda activate <environment_name>
  wandb --version
  ```
2. Create an account at [wandb.ai](https://wandb.ai/)
3. Authorize with your API key:
  ```
  wandb login
  ```
   
See [W&B Quickstart](https://wandb.ai/quickstart) for more details.

There are currently one implementation of the `Logger` class: `WandbLogger`. The logger can be replaced with another implementation later if desired. Within the configuration file, you can simply specify the package, module and the name of the new logger class. See [Config Documentation](config/README.md).

## Prepare Datasets
The LiDAR datasets were downloaded from [Norwegian Mapping Authority](https://hoydedata.no), and the cultural heritage database from [GeoNorge](https://geonorge.no) where used to annotate the datasets.
For more detailed information about preparing data: [Preprocessing in ArcGIS Pro](documentation/preproecessing.md).

##  Train
Using Slurm's job array structure, users can submit and run multiple instances of the same script independently. The following command creates an array of 4 jobs with index values of 0, 1, 2, and 3. The same job is executed with different configuration files using array jobs, which parallelize the computations. 
```
sbatch scripts/train.sh
```

See [https://slurm.schedmd.com/overview.html](https://slurm.schedmd.com/overview.html) for more information about using the slurm system. 

Alternatively run the python for a single configuration, without using slurm:
```
python src/train.py --config=<config_file.yml>
```
The argument(s) supported by the python script:

`--config`: Name of the configuration file, assuming it is located within the `config` folder. 
  Default: `config.yml`.

