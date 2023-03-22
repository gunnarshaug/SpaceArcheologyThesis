# Deep Learning with Airborne Laser Scanning
<!-- TABLE OF CONTENTS -->
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
    <li><a href="#demo">Demo</a></li>
    <li><a href="#train">Train</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


## About The Project


### Key Features



## Install Dependencies
<!-- you will need to install all the dependencies such as Pytorch, GDAL, etc., and also have GPU with CUDA driver working -->
### Prerequisites

### Installation
Here is an example to create an environment from scratch with `anaconda` or `miniconda` 
```
sbatch scripts/env_setup.sh
```
Alternatively:
```
conda env create --file environment.yml
```


## Demo

Run demo

```
python src/demo.py
```

##  Train
### Prepare Data

### Begin Training
Using slurm scripts:
```
sbatch scripts/faster_rcnn.sh
```

Alternatively: 
```
python src/train.py
```
Some key arguments:
- `--config`: Name of the configuration file, assuming it is located within the `config` folder. Default: `faster_rcnn.yml`.

## Previous Work

- [SpaceArcheology](https://github.com/arkadiy93/SpaceArcheologyThesis) 



