from torch.utils.data import DataLoader
from typing import Optional
import utils.general
import os
import pathlib

class DataLoaders:
    """
    DataLoaders used for object detection.
    """
    def __init__(self,
                  dir: str,
                  dataset_config: dict,
                  transform_opts: dict,
                  batch_size: int, 
                  num_workers: int):
        assert "width" in transform_opts
        assert "height" in transform_opts
        
        super().__init__()
        self.dir = pathlib.Path(dir)
        self.transform_opts = transform_opts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_object = utils.general.get_class(**dataset_config)

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
            
            self.train = self.dataset_object(
                root_dir=os.path.join(self.dir, "train"),
                transform=train_transform
            )
            
            self.val = self.dataset_object(
                root_dir=os.path.join(self.dir, "val"),
                transform=val_transform
            )
    
        if stage == "test" or stage is None:
            self.test = self.dataset_object(
                root_dir=os.path.join(self.dir, "test"),
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