import yaml
import os

class Dict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Dict(**value)
            else:
                self.__dict__[key] = value

def _get_cnf_path(relative_path):
    current_path = os.path.realpath(__file__)
    path = os.path.join(current_path,"..", "..", "..", relative_path)
    return os.path.abspath(path)

def load_cnf(config_path, is_absolute=False):
    if is_absolute == False:
        config_path = _get_cnf_path(config_path)
        
    with open(config_path) as file:
        config = yaml.safe_load(file)
        print(config)

    return Dict(**config)

load_cnf("t.cnf")