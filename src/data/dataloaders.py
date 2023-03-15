from torch.utils.data import random_split,  DataLoader
from typing import Optional
import data.datasets as module_data
import utils.general
import os


class DataLoaders:
    """
    DataLoaders used for object detection in space archeology
    """
    def __init__(self, 
                  data_dir: str,
                  dataset_type: str,
                  transform_opts: str,
                  batch_size: int, 
                  num_workers: int):
        assert "width" in transform_opts
        assert "height" in transform_opts
        
        super().__init__()
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.transform_opts = transform_opts
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """
        :param stage: Tells what stage the trainer has been called with.
        """
        val_transform = utils.general.get_transform(
            self.transform_opts, 
            is_train=False
        )
        
        if stage == "train" or stage is None:
            train_transform = utils.general.get_transform(
                self.transform_opts, 
                is_train=True
            )
            self.train = getattr(module_data, self.dataset_type)(
                root_dir=os.path.join(self.data_dir, "train"),
                transform=train_transform
            )
            self.val = getattr(module_data, self.dataset_type)(
                root_dir=os.path.join(self.data_dir, "val"),
                transform=val_transform
            )

            self.train.transform = train_transform
            self.val.transform = val_transform
    
        if stage == "test" or stage is None:
            self.test = getattr(module_data, self.dataset_type)(
                root_dir=os.path.join(self.data_dir, "test"),
                transform=val_transform
            )
            
    def train_dataloader(self):
        return DataLoader(dataset=self.train, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)
    def test_dataloader(self):
        return DataLoader(dataset=self.test, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)
        
def _collate_fn(batch) -> tuple:
    """
    copy from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
    """
    return tuple(zip(*batch))