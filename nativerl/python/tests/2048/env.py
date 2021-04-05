import math
from pathmind import pynativerl as nativerl
from pathmind.pynativerl import Continuous
from .base import Game2048


class Game2048Env(nativerl.Environment):

    def __init__(self, simulation=Game2048()):
        nativerl.Environment.__init__(self)
        self.simulation = simulation

    def getActionSpace(self, agent_id=0):
        return nativerl.Discrete(self.simulation.number_of_actions) if agent_id == 0 else None

    def getActionMaskSpace(self):
        return None

    def getObservationSpace(self):
        return nativerl.Continuous([-math.inf], [math.inf], [self.simulation.number_of_observations])

    def getNumberOfAgents(self):
        return 1

    def getActionMask(self, agent_id=0):
        return None

    def getObservation(self, agent_id=0):
        return nativerl.Array(self.simulation.get_observation())

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
        return self.simulation.get_reward()

    def getMetrics(self, agent_id=0):
        return nativerl.Array(self.simulation.get_metrics())

    def getMetricsSpace(self) -> Continuous:
        return nativerl.Continuous(low=[-math.inf], high=[math.inf], shape=[2])
