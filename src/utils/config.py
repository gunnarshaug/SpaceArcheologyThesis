import yaml
import os

class Dict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Dict(**value)
            else:
                self.__dict__[key] = value

def load(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
        print(config)
    config = Dict(**config)
    return config

if __name__ == "__main__":
    # print(os.path.abspath(CONFIG_PATH))
    cnf = load("faster_rcnn.yaml")
