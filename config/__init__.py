"""
Configuration module for Iris-Vein Biometrics
"""
import yaml
import os

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Export configuration
config = load_config()
