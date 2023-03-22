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
    <li><a href="#demo">Demo</a></li>
    <li><a href="#train">Train</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


## About The Project
The purpose of this study is to improve previous work conducted on detecting ancient burial mounds in Norway by utilizing a faster R-CNN object detection model with LiDAR-based Digital Terrain Models and identifying methods to reduce false positives.


## Install Dependencies
### Prerequisites

### Installation
Here is an example to create an environment from scratch with `anaconda` or `miniconda` 
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
The LiDAR datasets were downloaded from [Norwegian Mapping Authority](https://hoydedata.no), and the cultural heritage database from [GeoNorge](https://geonorge.no) where used to annotate the datasets.
For more detailed information about preparing data: [Preprocessing in ArcGIS Pro](docs/preproecessing.md).

### Begin Training

```
python src/train.py
```
Some key arguments:
- `--config`: Name of the configuration file, assuming it is located within the `config` folder. 
  Default: `faster_rcnn.yml`.



## Previous Work

- [SpaceArcheology](https://github.com/arkadiy93/SpaceArcheologyThesis) 



