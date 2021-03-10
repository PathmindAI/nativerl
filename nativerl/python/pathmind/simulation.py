from typing import List, Union, Dict, Optional
import math
import numpy as np

__all__ = ["Discrete", "Continuous", "Simulation"]


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


class Simulation:
    """Pathmind's Python interface for multiple agents. Make sure to initialize
    all parameters you need for your simulation here, so that e.g. the `reset`
    method can restart a new simulation."""

    action: Dict[int, Union[float, np.ndarray]] = None  # Dynamically generated for each state by Pathmind

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

    def action_space(self, agent_id: int) -> Union[Continuous, Discrete]:
        """Return a Discrete or Continuous action space per agent."""
        raise NotImplementedError

    def get_reward(self, agent_id: int) -> Dict[str, float]:
        """Get the reward terms of the simulation as a dictionary, given the current simulation state."""
        raise NotImplementedError

    def get_observation(self, agent_id: int) -> Dict[str, Union[float, List[float]]]:
        """Get a dictionary of observations for the current state of the simulation. Each
        observation can either be a """
        raise NotImplementedError

    def is_done(self, agent_id: int) -> bool:
        """Is the simulation over?"""
        raise NotImplementedError

    def get_metrics(self, agent_id: int) -> Optional[List[float]]:
        """Return a list of numerical values you want to track. If you don't
        specify any metrics, we simply use all provided observations for your agents."""
        return None
