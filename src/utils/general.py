import os
import csv
import yaml
import pathlib
import importlib
import torchvision
from torch.utils.data import ConcatDataset


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
    return config

def ensure_existing_dir(dirname: str) -> None:
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def get_class(package:str, module: str, name: str) -> any:
    """
    Returns the class object.
    :param package: the name of the package.
    :param module: the name of the module.
    :param name: name of the specific implementation class- should be located within the specified package and module.
    """
    try:
        module = importlib.import_module("{}.{}".format(package, module))
        return getattr(module, name)
    except Exception as ex:
        print(ex)
        raise Exception("Error while trying to find '{}' in module '{}'".format(
            name,
            module
        ))

def apply_nms(orig_prediction, iou_thresh: float = 0.3) -> dict:
  """
  NMS (Non-Maximum Suppression) - selects a singe box out of many overlapping boxes.
  :param iou_thresh: iou threshold
  :param orig_prediction: original prediction
  :returns the final iou threshold
  """
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep].cpu()
  final_prediction['scores'] = final_prediction['scores'][keep].cpu()
  final_prediction['labels'] = final_prediction['labels'][keep].cpu()
  
  return final_prediction

def generate_dataset(dirs: str, dataset_object,  transform):
    datasets = [dataset_object(root_dir=directory, transform=transform) for directory in dirs ]
    print([len(dataset) for dataset in datasets])

    return ConcatDataset(datasets)

def collate_fn(batch) -> tuple:
    """
    copy from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
    """
    return tuple(zip(*batch))