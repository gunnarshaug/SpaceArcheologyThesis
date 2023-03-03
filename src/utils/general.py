import os
import yaml
import datetime

class Dict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Dict(**value)
            else:
                self.__dict__[key] = value
        self.__dict__["timestamp"] = datetime.datetime.now().strftime('%d.%m.%Y_%H.%M.%S')


def _get_cfg_path(relative_path):
    current_path = os.path.realpath(__file__)
    path = os.path.join(current_path,"..", "..", "..", relative_path)
    return os.path.abspath(path)

def load_cfg(config_path, is_absolute=False):
    if is_absolute == False:
        config_path = _get_cfg_path(config_path)
        
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return Dict(**config)
