"""A function that loads configuration settings from a yaml file."""
import yaml

def load_config():
    file_path = 'config/config.yaml'
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config