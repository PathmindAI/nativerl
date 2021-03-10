import math
import typing
import itertools
import yaml
from collections import OrderedDict

from pathmind import pynativerl as nativerl
from pathmind.pynativerl import Continuous
from .mouse_env_pathmind import MouseAndCheese

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "obs.yaml"), "r") as f:
    schema: OrderedDict = yaml.safe_load(f.read())
OBS = schema.get("observations")


class MouseEnv(nativerl.Environment):

    def __init__(self, simulation=MouseAndCheese()):
        nativerl.Environment.__init__(self)
        self.simulation = simulation

    def getActionSpace(self, agent_id=0):
        space = self.simulation.action_space(agent_id=agent_id)
        if hasattr(space, "choices"):  # Discrete space defined
            nativerl_space = nativerl.Discrete(n=space.choices, size=space.size)
        else:  # Continuous space defined
            nativerl_space = nativerl.Continuous(low=[space.low], high=[space.high], shape=space.shape)
        return nativerl_space if agent_id < self.getNumberOfAgents() else None

    def getObservationSpace(self):
        obs_shape = [len(self.getObservation(agent_id=0))]
        return nativerl.Continuous([-math.inf], [math.inf], obs_shape)

    def getNumberOfAgents(self):
        return self.simulation.number_of_agents()

    def getActionMask(self, agent_id=0):
        return None

    def getActionMaskSpace(self):
        return None

    def getObservation(self, agent_id=0):
        obs_dict = self.simulation.get_observation(agent_id)

        lists = [[obs_dict[obs]] if not isinstance(obs_dict[obs], typing.List) else obs_dict[obs] for obs in OBS]
        observations = list(itertools.chain(*lists))

        return nativerl.Array(observations)

    def reset(self):
        self.simulation.reset()

    def setNextAction(self, action, agent_id=0):
        if not self.simulation.action:
            self.simulation.action = {}
        self.simulation.action[str(agent_id)] = action

    def isSkip(self, agent_id=0):
        return False

    def step(self):
        return self.simulation.step()

    def isDone(self, agent_id=0):
        return self.simulation.is_done(agent_id)

    def getReward(self, agent_id=0):
        # TODO: if reward snippet, call it here
        reward_sum = sum(self.simulation.get_reward(agent_id).values())
        return reward_sum

    def getMetrics(self, agent_id=0):
        if self.simulation.get_metrics(agent_id):
            return self.simulation.get_metrics(agent_id)
        else:
            return list(self.simulation.get_observation(agent_id).values())

    def getMetricsSpace(self) -> Continuous:
        num_metrics = len(self.getMetrics())
        return nativerl.Continuous(low=[-math.inf], high=[math.inf], shape=[num_metrics])
