import yaml
from os.path import dirname, abspath, join
import sys

parent_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_path)

def load_config(config_path):
    with open(join(config_path,"config.yaml"), 'r') as file:
        return yaml.safe_load(file)
