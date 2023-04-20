from torch.utils.data import DataLoader, ConcatDataset, Dataset
import utils.general
import albumentations as alb
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
                  num_workers: int=0):
        error_msg = "[!] 'dataset_config' should be a dictionary containing the keys 'package', 'module' and 'name'"
        assert "package" in dataset_config, error_msg
        assert "module" in dataset_config, error_msg
        assert "name" in dataset_config, error_msg
        self.dataset_object = utils.general.get_class(**dataset_config)
        
        self.transform_opts = transform_opts
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self._train = self._generate_dataset(train_dirs, True)
        self._val = self._generate_dataset(val_dirs, False)
        self._test = self._generate_dataset(test_dirs, False)
    
                
    def _generate_dataset(self, dirs: list, is_train: bool) -> Dataset:
        transform = _get_transform(
            self.transform_opts, 
            is_train=is_train
        )
        assert len(dirs) > 0
        dataset = utils.general.generate_dataset(dirs, self.dataset_object, transform)
        print(dirs)
        assert len(dataset) > 0, "[!] Dataset cannot be empty"
        return dataset
    
    @property
    def test_length(self):
        error_msg = "[!] The test dataset is not properly configured."
        assert self._test is not None, error_msg
        return len(self._test)
    
    @property
    def train_length(self):
        error_msg = "[!] The train dataset is not properly configured."
        assert self._train is not None, error_msg
        return len(self._train)
    @property
    def val_length(self):
        error_msg = "[!] The validation dataset is not properly configured."
        assert self._val is not None, error_msg
        return len(self._val)

    def generate_train_dataloader(self) -> DataLoader:
        error_msg = "[!] The train dataset is not properly configured."
        assert self._train is not None, error_msg
        return DataLoader(
                dataset=self._train, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                collate_fn=utils.general.collate_fn,
                pin_memory=True,
                shuffle=True
            )

    def generate_val_dataloader(self) -> DataLoader:
        error_msg = "[!] The validation dataset is not properly configured."
        assert self._val is not None, error_msg
        
        return DataLoader(
                dataset=self._val, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                collate_fn=utils.general.collate_fn,
                pin_memory=True,
                shuffle=True)

        
    def generate_test_dataloader(self) -> DataLoader:
        error_msg = "[!] The test dataset is not properly configured."
        assert self._test is not None, error_msg
        
        return DataLoader(dataset=self._test, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=utils.general.collate_fn,
                          pin_memory=True,
                          shuffle=True)
        
def _get_transform(options: dict, is_train:bool=False) -> alb.Compose:
    error_msg = "[!] transform options 'options' should be a dictionary containing the keys 'width' and 'height'"
    assert "width" in options, error_msg
    assert "height" in options, error_msg
    
    transforms = []
    transforms.append(alb.Resize(options["width"], options["height"]))

    if is_train:
        transforms.append(alb.HorizontalFlip(p=0.5))
        transforms.append(alb.RandomBrightnessContrast(p=0.2))
        
    mean = options.get("mean", [0.5, 0.5, 0.5])
    std = options.get("std", [0.5, 0.5, 0.5])
    print(mean,std)

    transforms.append(
        alb.Normalize(mean=mean, std=std)
    )
    
    transforms.append(alb.pytorch.transforms.ToTensorV2())

    return alb.Compose(
        transforms,
        bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['class_labels'])
    )
