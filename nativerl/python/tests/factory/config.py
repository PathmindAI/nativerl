import os
import yaml


full_dir_name = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(full_dir_name, "./config.yml")

with open(config_file_path, "r") as f:
    config = yaml.safe_load(f.read()).get("config")

SIMULATION_CONFIG = config
MASK_KEY = "action_mask"
OBS_KEY = "observations"


def get_observation_names():
    return [k for k, v in SIMULATION_CONFIG.items() if k.startswith('obs_') and v is True]


def get_reward_names_and_weights():
    return {k: v.get("weight") for k, v in SIMULATION_CONFIG.items() if k.startswith('rew_') and v.get("value") is True}
