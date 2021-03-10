import typing
from typing import List
import math


class Discrete:
    """A discrete action space of given size, with the specified number of choices.

    For instance, a Discrete(2) corresponds to a binary choice (0 or 1),
    a Discrete(10) corresponds to an action space with 10 discrete options (0 to 9)
    and a Discrete(3, 2) represents vectors of length two, each with 3 choices, so
    a valid choice would be [0, 1] or [2, 2].

    """
    def __init__(self, choices: int, size: int = 1):
        self.choices = choices
        self.size = size


class Continuous:
    """An action space with continuous values of given shape with specified
    value ranges between "low" and "high".

    For instance, a Continuous([3], 0, 1) has length 3 vectors with values in
    the interval [0, 1] each, whereas a Continuous([3, 2]) accepts values of
    shape (3,2).

    """
    def __init__(self, shape: List[int], low: float = -math.inf, high: float = math.inf):
        self.shape = shape
        self.low = low
        self.high = high


class SingleAgentSimulation:
    """Pathmind's Python interface for single agents."""

    # TODO: how to make sure this is used correctly by users?
    action = None  # Dynamically generated for each state by Pathmind

    def __init__(self, *args, **kwargs):
        """Set any properties and initial states needed for your simulation.Make sure to initialize
        all parameters you need for your simulation here, so that e.g. the `reset`
        method can restart a new simulation."""
        pass

    def step(self) -> None:
        """Carry out all things necessary at the next time-step of your simulation,
        in particular update the state of it. You have access to 'self.action' from
        Pathmind's backend."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset your simulation parameters."""
        raise NotImplementedError

    def action_space(self) -> typing.Union[Continuous, Discrete]:
        """Return a Discrete or Continuous action space"""
        raise NotImplementedError

    def get_reward(self) -> typing.Dict[str, float]:
        """Get the reward terms of the simulation as a dictionary, given the current simulation state."""
        raise NotImplementedError

    def get_observation(self) -> typing.Dict[str, typing.Union[float, List[float]]]:
        """Get a dictionary of observations for the current state of the simulation. Each
        observation can either be a """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Is the simulation over?"""
        raise NotImplementedError

    def get_metrics(self) -> typing.Optional[typing.List[float]]:
        """Return a list of numerical values you want to track. If you don't
        specify any metrics, we simply use all provided observations for your agent."""
        return None


class MultiAgentSimulation:
    """Pathmind's Python interface for multiple agents. Make sure to initialize
    all parameters you need for your simulation here, so that e.g. the `reset`
    method can restart a new simulation."""

    # TODO: how to make sure this is used correctly by users?
    action = None  # Dynamically generated for each state by Pathmind

    def __init__(self, *args, **kwargs):
        """Set any properties and initial states needed for your simulation."""
        pass

    def step(self) -> None:
        """Carry out all things necessary at the next time-step of your simulation,
        in particular update the state of it. You have access to 'self.action' from
        Pathmind's backend."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset your simulation parameters."""
        raise NotImplementedError

    def number_of_agents(self) -> int:
        """Returns the total number of agents to be controlled by Pathmind."""
        raise NotImplementedError

    def action_space(self, agent_id: int) -> typing.Union[Continuous, Discrete]:
        """Return a Discrete or Continuous action space per agent."""
        raise NotImplementedError

    def get_reward(self, agent_id: int) -> typing.Dict[str, float]:
        """Get the reward terms of the simulation as a dictionary, given the current simulation state."""
        raise NotImplementedError

    def get_observation(self, agent_id: int) -> typing.Dict[str, typing.Union[float, List[float]]]:
        """Get a dictionary of observations for the current state of the simulation. Each
        observation can either be a """
        raise NotImplementedError

    def is_done(self, agent_id: int) -> bool:
        """Is the simulation over?"""
        raise NotImplementedError

    def get_metrics(self, agent_id: int) -> typing.Optional[typing.List[float]]:
        """Return a list of numerical values you want to track. If you don't
        specify any metrics, we simply use all provided observations for your agent."""
        return None
