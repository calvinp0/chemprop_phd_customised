import os
import yaml

from order_reactions import parse_and_order_reactions
from connect_reactions import connect_reactions
from create_sdf import create_sdf
from create_target import create_target

def load_config(config_path: str = "pipeline_config.yaml") -> dict:
    """Load YAML config and return it as a dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():

    config = load_config("/home/calvin/code/chemprop_phd_customised/habnet/data/preprocessing/pipeline_config.yml")

    parse_and_order_args = config["parse_and_order_reactions"]
    connect_args = config["connect_reactions"]
    create_sdf_args = config["create_sdf"]
    create_target_args = config["create_target"]

    parse_and_order_reactions(**parse_and_order_args)
    connect_reactions(**connect_args)
    create_sdf(**create_sdf_args)
    create_target(**create_target_args)

if __name__ == '__main__':
    main()
