import os
import yaml
import datetime
from pathlib import Path
import albumentations as a
import albumentations.pytorch.transforms
import os
import logging


def setup_logging():
  logging.basicConfig(
      level=logging.DEBUG,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[
          logging.FileHandler("debug.log"),
          logging.StreamHandler()
      ]
  )
def load_yaml(path):
    with open(path) as file:
        return yaml.safe_load(file)

def _get_cfg_path(relative_path):
    current_path = os.path.realpath(__file__)
    path = os.path.join(current_path,"..", "..", "..", relative_path)
    return os.path.abspath(path)

def load_cfg(config_path, is_absolute=False) -> dict:
    if is_absolute == False:
        config_path = _get_cfg_path(config_path)
        
    config = load_yaml(config_path)
    config['timestamp'] = datetime.datetime.now().strftime('%d.%m.%Y_%H.%M.%S')
    return config
    # return Dict(**config)


def ensure_existing_dir(dirname: str) -> None:
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)



def get_transform(dimensions: dict, is_train:bool=False) -> a.Compose:
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
