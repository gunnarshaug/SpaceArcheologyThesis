from torch.utils.data import DataLoader, ConcatDataset, Dataset
import utils.general
import pathlib 
import albumentations as a
import albumentations.pytorch.transforms

class DataLoaders:
    """
    DataLoaders used for object detection.
    :param train_dirs: list of the directories used for training. 
    :param val_dirs: list of the directories used for validating. 
    :param test_dirs: list of the directories used for testing. 
    :param dataset_config: a dictionary that contains the implementation details of the dataset.
    :param transform_opts: a dictionary that contains the width and height of the transformed images.
    :param batch_size
    :param num_workers
    :param val_size: validation size used to sample the training data into training and validation.
    """
    def __init__(self,
                  train_dirs: list,
                  val_dirs: list,
                  test_dirs: list,
                  dataset_config: dict,
                  transform_opts: dict,
                  batch_size: int, 
                  num_workers: int):
        error_msg = "[!] 'transform_opts' should be a dictionary containing the keys 'width' and 'height'"
        assert "width" in transform_opts, error_msg
        assert "height" in transform_opts, error_msg
                
        error_msg = "[!] 'dataset_config' should be a dictionary containing the keys 'package', 'module' and 'name'"
        assert "package" in dataset_config, error_msg
        assert "module" in dataset_config, error_msg
        assert "name" in dataset_config, error_msg

        self.transform_opts = transform_opts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_object = utils.general.get_class(**dataset_config)
        
        self.train = self._generate_dataset(train_dirs, True)
        self.val = self._generate_dataset(val_dirs, False)
        self.test = self._generate_dataset(test_dirs, False)

                
    def _generate_dataset(self, dirs: list, is_train: bool) -> Dataset:
        transform = _get_transform(
            self.transform_opts, 
            is_train=is_train
        )
        
        datasets = [self.dataset_object(root_dir=pathlib.Path(directory), transform=transform) for directory in dirs ]
            
        return ConcatDataset(datasets)

    @property
    def train_dataloader(self):
        error_msg = "[!] The train dataset is not properly configured."
        assert self.train is not None, error_msg
        
        return DataLoader(dataset=self.train, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)

    @property
    def val_dataloader(self):
        error_msg = "[!] The validation dataset is not properly configured."
        assert self.val is not None, error_msg
        
        return DataLoader(dataset=self.val, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=_collate_fn)
        
    @property
    def test_dataloader(self):
        error_msg = "[!] The test dataset is not properly configured."
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


def _get_transform(dimensions: dict, is_train:bool=False) -> a.Compose:
    transforms = []
    transforms.append(a.Resize(dimensions["width"], dimensions["height"]))

    if is_train:
        transforms.append(a.HorizontalFlip(p=0.5))
        transforms.append(a.RandomBrightnessContrast(p=0.2))

    transforms.append(
        a.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    transforms.append(albumentations.pytorch.transforms.ToTensorV2())

    return a.Compose(
        transforms,
        bbox_params=a.BboxParams(format='pascal_voc', label_fields=['class_labels'])
    )
