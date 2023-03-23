# Configuration Files
Configuration files are used to store key value pairs or some configurable information that could be read or accessed in the code. It makes the settings and code more reusable and keep the configuration information at a centralized location and segregated.

## Training
### Format
The following explains the format of the configuration file used for training.

* `experiment`: Contains information about a specific experiment. This will be used with the logger to create useful naming conventions. 
  - `name`: Name of the experiment. Be consistent with the naming conventions to easily track each experiments. With wandb, the `name` will be the name of a specific project. If multiple experiments have the same name, they will appear within the same project wandb-dashboard and can be easily compared with each other. Each run/execution within an experiment are named by date followed by `preprocessing`.
  - `preprocessing`: Short description about what processing steps has been done to the dataset. For example SLRM or SLOPE. 
  - `description`: Some notes to remember about the project.
  - `tags`: Specifically for wandb logging. Can be used to search/filter in the dashboard. 

* `data`: contains information about where the datasets are located.
  - `train_dirs`: The path to the directories where the datasets are located. 
  - `test_dirs`: The path to the directories where the datasets are located. 
  
* `dataloader`: contains information to configure dataloaders batch_size, num_workers and transform options.
  - `batch_size`
  - `num_workers`
  - `transform_opts`: includes the `width` and `height` of the transformed images. 

* `training`: Contains information related to training a model (hyperparameters)
  - `seed` (optional): Used to support reproducibility. See [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html). Default value is 1.
  - `epochs`
  - `optimizer`: The current implementation supports stocastic gradient decent. 
  - `scheduler`: The current implementation supports steplr
  
* `classes`: Specific implementations for a class. It is required to specify what `Dataset` and `Logger` implementations are used.  
  - `name`: Name of the class.
  - `package`: Name of the python package.
  - `module`: Name of the python module.



### Example
Example using `PascalVOCDataset` and `WandbLogger` as specific implementation classes. Remember to specify what package and module they belong to.
```
classes:
  dataset: 
    name: Dataset2022
    package: data
    module: datasets
  logger: 
    name: WandbLogger
    package: loggers
    module: wandb

data:
  test_dirs: 
    - replication/data
  train_dirs:
    - replication/data

dataloader:
  batch_size: 10
  num_workers: 4
  transform_opts: 
    width: 224
    height: 224

training:
  seed: 1
  epochs: 1
  optimizer:    
    type: sdg
    lr: 0.005
    momentum: 0.9
    weight_decay: 0.0005
  scheduler: 
    type: steplr
    step_size: 30
    gamma: 0.1
```