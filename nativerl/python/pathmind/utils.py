from .simulation import Simulation
import os
import yaml


def write_observation_yaml(simulation: Simulation, file_path) -> None:
    obs_name_list = list(simulation.get_observation(0).keys())
    obs = {"observations": obs_name_list}
    with open(os.path.join(file_path, "obs.yaml"), "w") as f:
        f.write(yaml.dump(obs))
