from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import utils.general
import pathlib
import numpy as np

class DataLoaders:
    """
    DataLoaders used for object detection.
    :param dataset_config: a dictionary that contains the implementation details of the dataset.
    :param transform_opts: a dictionary that contains the width and height of the transformed images.
    :param batch_size
    :param num_workers
    :param val_size: validation size used to sample the training data into training and validation.
    :param random_seed: seed for reproducibility
    """
    def __init__(self,
                  dataset_config: dict,
                  transform_opts: dict,
                  batch_size: int, 
                  num_workers: int,
                  val_size: float=0.1,
                  random_seed: int=1):
        error_msg = "[!] 'transform_opts' should be a dictionary containing the keys 'width' and 'height'"
        assert "width" in transform_opts, error_msg
        assert "height" in transform_opts, error_msg
        
        error_msg = "[!] 'val_size' should be in the range [0, 1]."
        assert ((val_size >= 0) and (val_size <= 1)), error_msg
        
        error_msg = "[!] 'dataset_config' should be a dictionary containing the keys 'package', 'module' and 'name'"
        assert "package" in dataset_config, error_msg
        assert "module" in dataset_config, error_msg
        assert "name" in dataset_config, error_msg
        
        super().__init__()
        self.transform_opts = transform_opts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_object = utils.general.get_class(**dataset_config)
        self.val_size = val_size
        self.random_seed = random_seed

    def setup(self, data_dir:str, stage: str="train"):
        """
        Configure the datasets for a specific stage.
        :param data_dir: The location of the training or testing data. 
        :param stage: Tells wither to setup training data or testing data. 
        """
        assert stage in ("train", "test")
        
        if stage == "train":
            self._setup_training_data(data_dir)
    
        if stage == "test":
            self._setup_test_data(data_dir)

    def _setup_test_data(self, dir):
        transform = utils.general.get_transform(
            self.transform_opts, 
            is_train=False
        )
        self.test = self.dataset_object(
            root_dir=dir,
            transform=transform
        )

    def _setup_training_data(self, dir:str):
        val_transform = utils.general.get_transform(
            self.transform_opts, 
            is_train=False
        )
        
        train_transform = utils.general.get_transform(
            self.transform_opts, 
            is_train=True
        )
        
        self.train = self.dataset_object(
            root_dir=dir,
            transform=train_transform
        )
        
        self.val = self.dataset_object(
            root_dir=dir,
            transform=val_transform
        )
        
        # setting up a random sampler
        num_train = len(self.train)
        indices = list(range(num_train))
        split = int(np.floor(self.val_size * num_train))
        
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(valid_idx)
        
    @property
    def train_dataloader(self):
        error_msg = "[!] The train dataset is not properly configured. Make sure to use the setup method with mode='train'"
        assert self.train is not None, error_msg
        
        return DataLoader(dataset=self.train, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          sampler=self.train_sampler,
                          collate_fn=_collate_fn)
    @property
    def val_dataloader(self):
        error_msg = "[!] The validation dataset is not properly configured. Make sure to use the setup method with mode='train'"
        assert self.val is not None, error_msg
        
        return DataLoader(dataset=self.val, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          sampler=self.val_sampler,
                          collate_fn=_collate_fn)
        
    @property
    def test_dataloader(self):
        error_msg = "[!] The test dataset is not properly configured. Make sure to use the setup method with mode='test'"
        assert self.test is not None, error_msg
        
        return DataLoader(dataset=self.test, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)
        
def _collate_fn(batch) -> tuple:
    """
    copy from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
    """
    return tuple(zip(*batch))