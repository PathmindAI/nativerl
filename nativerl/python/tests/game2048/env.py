import math
import typing
import itertools
import yaml
import numpy as np
from collections import OrderedDict

from pathmind_training import pynativerl as nativerl
from pathmind_training.pynativerl import Continuous
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

    def getObservationSpace(self):
        obs_shape = [self.simulation.number_of_observations]
        return nativerl.Continuous([-math.inf], [math.inf], obs_shape)

    def getNumberOfAgents(self):
        return 1

    def getActionMask(self, agent_id=0):
        return None

    def getActionMaskSpace(self):
        return None

    def getObservation(self, agent_id=0):
        obs_dict = self.simulation.get_observation()

        lists = [[obs_dict[obs]] if not isinstance(obs_dict[obs], typing.List) else obs_dict[obs] for obs in OBS]
        observations = list(itertools.chain(*lists))

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
        if self.simulation.get_metrics():
            return self.simulation.get_metrics()
        else:
            return list(self.simulation.get_observation().values())

    def getMetricsSpace(self) -> Continuous:
        num_metrics = len(self.getMetrics())
        return nativerl.Continuous(low=[-math.inf], high=[math.inf], shape=[num_metrics])
