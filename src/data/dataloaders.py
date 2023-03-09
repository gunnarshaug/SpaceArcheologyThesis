from .base_dataloader import BaseDataLoader
import data.datasets as module_data
import utils.general
import os

def _collate_fn(batch) -> tuple:
    """
    copy from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
    """
    return tuple(zip(*batch))

class MainDataLoader(BaseDataLoader):
    def __init__(self, 
                  data_dir: str,
                  dataset_type: str,
                  transform_opts: str,
                  batch_size: int, 
                  num_workers: int,
                  mode: str="train",
                  collate_fn=_collate_fn
                ):
        assert mode in ("train", "val", "test")
        assert "width" in transform_opts
        assert "height" in transform_opts

        self.data_dir = data_dir
        is_train = mode == "train"
        transform = utils.general.get_transform(transform_opts, is_train)
        self.dataset = getattr(module_data, dataset_type)(
            os.path.join(self.data_dir, mode),
            transform
        )
        super().__init__(self.dataset, batch_size, num_workers, collate_fn)

