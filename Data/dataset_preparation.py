import os
import yaml
from pathlib import Path

from data_extraction_openCARP import data_extracrion_openCARP
from data_constructor_openCARP import data_constructor_openCARP


class Args:
    def __init__(self, config_path):
        self.config_path = config_path

        with open(self.config_path, 'r') as f:
            self.config_yaml = yaml.safe_load(f)
           
        for key, value in self.config_yaml.items(): 
            setattr(self, key, value)

        self.folder_name = self.sim_folder

if __name__ == '__main__':
    config_path = Path(__file__).resolve().with_name('config.yaml')
    args = Args(str(config_path))
    data_extracrion_openCARP(args)
    data_constructor_openCARP(args)
