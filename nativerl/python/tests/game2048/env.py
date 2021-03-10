import math
import typing
import itertools
import yaml
import numpy as np
from collections import OrderedDict

from pathmind import pynativerl as nativerl
from pathmind.pynativerl import Continuous
from .base import Game2048

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "obs.yaml"), "r") as f:
    schema: OrderedDict = yaml.safe_load(f.read())
OBS = schema.get("observations")


class Game2048Env(nativerl.Environment):

    def __init__(self, simulation=Game2048()):
        nativerl.Environment.__init__(self)
        self.simulation = simulation

    def getActionSpace(self, agent_id=0):
        return nativerl.Discrete(self.simulation.number_of_actions) if agent_id == 0 else None

    def getActionMaskSpace(self):
        return None

    def getObservationSpace(self):
        if hasattr(self.simulation, "observation_shape"):
            obs_shape = self.simulation.observation_shape
        else:
            obs_shape = [self.simulation.number_of_observations]

        return nativerl.Continuous([-math.inf], [math.inf], obs_shape)

    def getNumberOfAgents(self):
        return 1

    def getActionMask(self, agent_id=0):
        return None

    def getObservation(self, agent_id=0):
        obs_dict = self.simulation.get_observation()
        is_numpy = type(list(obs_dict.values())[0]) == np.ndarray
        if not is_numpy:
            lists = [[obs_dict[obs]] if not isinstance(obs_dict[obs], typing.List) else obs_dict[obs] for obs in OBS]
            observations = list(itertools.chain(*lists))
        else:
            observations = np.concatenate(list(obs_dict.values())).reshape(self.simulation.observation_shape)
        return nativerl.Array(observations)

    def reset(self):
        self.simulation.reset()

    def setNextAction(self, action, agent_id=0):
        self.simulation.action = action

    def isSkip(self, agent_id=0):
        return False

    def step(self):
        return self.simulation.step()

    def isDone(self, agent_id=0):
        return self.simulation.is_done()

    def getReward(self, agent_id=0):
        # TODO: if reward snippet, call it here
        reward_sum = sum(self.simulation.get_reward().values())
        return reward_sum

    def getMetrics(self, agent_id=0):
        return nativerl.Array(self.simulation.get_metrics())

    def getMetricsSpace(self) -> Continuous:
        return nativerl.Continuous(low=[-math.inf], high=[math.inf], shape=[self.simulation.number_of_metrics])
