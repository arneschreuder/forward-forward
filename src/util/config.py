import importlib

import yaml


class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(v)
            else:
                self.__dict__[k] = v

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self) -> str:
        return str(self.__dict__)


def load_config(file_path):
    with open(file_path, "r") as stream:
        data_loaded = yaml.safe_load(stream)
    return Config(data_loaded)


def load_class(path):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    return class_


def instantiate_class_from_config(config_item):
    cls = load_class(config_item.path)
    if hasattr(config_item, "params"):  # Check if parameters exist
        params = {
            k: v if not isinstance(v, Config) else instantiate_class_from_config(v)
            for k, v in config_item.params.__dict__.items()
        }
        if "optim" in config_item.path.lower():  # Return class if it's an optimizer
            return cls
        return cls(**params)
    else:  # If no parameters, return class directly
        if "optim" in config_item.path.lower():  # Return class if it's an optimizer
            return cls
        return cls()
