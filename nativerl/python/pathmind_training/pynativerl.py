import importlib
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np


def init(args):
    pass


class Space:
    pass


class Discrete(Space):
    def __init__(self, n: int, size: int = 1):
        self.n = n
        self.size = size


class Continuous(Space):
    def __init__(self, low: List[float], high: List[float], shape: List[int]):
        self.low = low
        self.high = high
        self.shape = shape


# Smart hack: use a pass-through function to act as Array constructor (already have numpy)
def Array(arr: Union[np.array, List]):
    return np.asarray(arr)


class Environment(ABC):
    @abstractmethod
    def getActionSpace(self, agent_id: int = 0) -> Optional[Space]:
        return NotImplemented

    # TODO: why is this not per agent if action space is?
    @abstractmethod
    def getActionMaskSpace(self) -> Continuous:
        return NotImplemented

    # TODO: Going forward this should be per agent, too
    @abstractmethod
    def getObservationSpace(self) -> Continuous:
        return NotImplemented

    @abstractmethod
    def getMetricsSpace(self) -> Continuous:
        return NotImplemented

    @abstractmethod
    def getNumberOfAgents(self) -> int:
        return NotImplemented

    @abstractmethod
    def getActionMask(self, agent_id: int = 0) -> Array:
        return NotImplemented

    @abstractmethod
    def getObservation(self, agent_id: int = 0) -> Array:
        return NotImplemented

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def setNextAction(self, action: Array, agent_id: int = 0) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def isSkip(self, agent_id: int = 0) -> bool:
        return NotImplemented

    @abstractmethod
    def isDone(self, agent_id: int = 0) -> bool:
        return NotImplemented

    @abstractmethod
    def getReward(self, agent_id: int = 0) -> float:
        return NotImplemented

    @abstractmethod
    def getMetrics(self, agent_id: int = 0) -> Array:
        return NotImplemented

    @abstractmethod
    def getRewardTerms(self, agent_id: int = 0) -> Array:
        return NotImplemented


def get_environment_class(env_name):
    """Get environment class instance from a string, interpreted as Python module
    :param env_name:
    :return:
    """
    class_name = env_name.split(".")[-1]
    module = env_name.replace(f".{class_name}", "")
    lib = importlib.import_module(module)
    return getattr(lib, class_name)


def createEnvironment(env_name):
    clazz = get_environment_class(env_name)
    obj = clazz()
    return obj
