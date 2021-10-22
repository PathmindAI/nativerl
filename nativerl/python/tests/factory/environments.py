"""In which environment are things happening? (where)
The environment specifies what agents can observe and how
they are rewarded for actions."""
import pprint

from .config import MASK_KEY, OBS_KEY, SIMULATION_CONFIG
from .controls import Action, do_action
from .features import *
from .models import Direction
from .util import factory_string, print_factory
from .util.samples import factory_from_config

PRINTER = pprint.PrettyPrinter(indent=2)

import importlib
from copy import deepcopy
from typing import Dict

import gym
import numpy as np
import ray
from gym import spaces
from ray import rllib

__all__ = [
    "FactoryEnv",
    "RoundRobinFactoryEnv",
    "MultiAgentFactoryEnv",
    "TupleFactoryEnv",
    "register_env_from_config",
    "get_observation_space",
    "get_action_space",
    "add_masking",
]


def register_env_from_config():
    env = SIMULATION_CONFIG.get("env")
    cls = getattr(importlib.import_module("factory.environments"), env)
    ray.tune.registry.register_env("factory", lambda _: cls())


def add_masking(self, observations):
    """Add masking, if configured, otherwise return observations as they were."""
    if self.masking:
        if self.config.get("env") == "MultiAgentFactoryEnv":  # Multi-agent scenario
            for key, obs in observations.items():
                observations[key] = {
                    MASK_KEY: update_action_mask(self, agent=key),
                    OBS_KEY: obs,
                }
        else:
            observations = {
                MASK_KEY: update_action_mask(self),
                OBS_KEY: observations,
            }
    return observations


def update_action_mask(env, agent=None):
    if agent is not None:
        current_agent = agent
    else:
        current_agent = env.current_agent
    agent_table = env.factory.tables[current_agent]
    agent_node = agent_table.node

    can_move_up = can_move_in_direction(agent_node, Direction.up, env.factory)
    can_move_right = can_move_in_direction(agent_node, Direction.right, env.factory)
    can_move_down = can_move_in_direction(agent_node, Direction.down, env.factory)
    can_move_left = can_move_in_direction(agent_node, Direction.left, env.factory)

    can_move_at_all = any([can_move_up, can_move_right, can_move_down, can_move_left])

    return np.array(
        [
            # Mask out illegal moves and collisions (no auto-regression, collisions still possible in MultiEnv)
            can_move_up,
            can_move_right,
            can_move_down,
            can_move_left,
            not can_move_at_all,  # Only allow not to move if no other move is valid
            # not agent_table.has_core(),  # Allow not moving only if table has no core anymore
        ]
    )


def get_observation_space(config, factory=None) -> spaces.Space:
    if not factory:
        factory = factory_from_config(config)
    dummy_obs = get_observations(0, factory)

    masking = config.get("masking")
    num_actions = config.get("actions")

    observation_space = spaces.Box(
        low=config.get("low"),
        high=config.get("high"),
        shape=(len(dummy_obs),),
        dtype=np.float32,
    )

    if masking:  # add masking
        observation_space = spaces.Dict(
            {
                MASK_KEY: spaces.Box(0, 1, shape=(num_actions,)),
                OBS_KEY: observation_space,
            }
        )
    return observation_space


def get_action_space(config):
    return spaces.Discrete(config.get("actions"))


def get_tuple_action_space(config):
    num_actions = config.get("actions")
    agents = config.get("num_tables")
    return spaces.Tuple([spaces.Discrete(num_actions) for _ in range(agents)])


class FactoryEnv(gym.Env):
    """Define a simple OpenAI Gym environment for a single agent."""

    metadata = {"render.modes": ["human", "ansi", "debug"]}

    def __init__(self, config=None):
        if config is None:
            config = SIMULATION_CONFIG
        self.config = config
        self.factory = factory_from_config(config)
        self.initial_factory = deepcopy(self.factory)
        self.num_agents = self.config.get("num_tables")
        self.num_actions = self.config.get("actions")
        self.masking = self.config.get("masking")
        self.action_mask = None
        self.current_agent = 0
        self.num_episodes = 0

        self.action_space = get_action_space(self.config)
        self.observation_space = get_observation_space(self.config, self.factory)

    def _step_apply(self, action):
        assert action in range(self.num_actions)

        table = self.factory.tables[self.current_agent]
        action_result = do_action(table, self.factory, Action(action))
        self.factory.add_move(self.current_agent, Action(action), action_result)

    def _step_observe(self):
        observations: np.ndarray = get_observations(self.current_agent, self.factory)
        rewards = get_reward(self.current_agent, self.factory, self.num_episodes)
        done = self._done()
        if done:
            self.factory.add_completed_step_count()
            self.num_episodes += 1
            self.factory.record_stats()
            self.factory.print_stats(self.num_episodes)

        observations = add_masking(self, observations)

        return observations, rewards, done, {}

    def _done(self):
        return get_done(self.current_agent, self.factory)

    def step(self, action):
        self._step_apply(action)
        return self._step_observe()

    def render(self, mode="debug"):
        if mode == "ansi":
            return factory_string(self.factory)
        elif mode == "human":
            return print_factory(self.factory)
        elif mode == "debug":
            return print_factory(self.factory, clear=False)
        else:
            super(self.__class__, self).render(mode=mode)

    def _reset(self):
        if self.config.get("random_init"):
            self.factory = factory_from_config(self.config)
        else:
            self.factory = deepcopy(self.initial_factory)

        self.render()
        observations = get_observations(self.current_agent, self.factory)
        observations = add_masking(self, observations)
        return observations

    def reset(self):
        return self._reset()


class TupleFactoryEnv(FactoryEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self.action_space = get_tuple_action_space(self.config)
        # All agents active at once. We set this as safeguard to prevent the addition
        # of agent-specific features by accident.
        self.current_agent = None

    def _step_apply(self, action_tuple):
        """Just go through all actions in the tuple and apply the moves as before individually."""
        for index, action in enumerate(action_tuple):
            assert action in range(self.num_actions)

            table = self.factory.tables[index]
            action_result = do_action(table, self.factory, Action(action))
            self.factory.add_move(index, Action(action), action_result)

    def _step_observe(self):
        observations: np.ndarray = get_observations(self.current_agent, self.factory)
        rewards = sum(
            [
                get_reward(index, self.factory, self.num_episodes)
                for index in range(self.num_agents)
            ]
        )

        done = self._done()
        if done:
            self.factory.add_completed_step_count()
            self.num_episodes += 1
            self.factory.print_stats(self.num_episodes)
            self.factory.record_stats()

        observations = add_masking(self, observations)

        return observations, rewards, done, {}

    def _done(self):
        return all(get_done(agent, self.factory) for agent in range(self.num_agents))


class RoundRobinFactoryEnv(FactoryEnv):
    def __init__(self, config=None):
        super().__init__(config)

    def step(self, action):
        """Cycle through agents, all else remains the same"""
        self._step_apply(action)
        self.current_agent = (self.current_agent + 1) % self.num_agents
        return self._step_observe()

    def _done(self):
        return all(get_done(agent, self.factory) for agent in range(self.num_agents))

    def reset(self):
        self.current_agent = 0
        return self._reset()


class MultiAgentFactoryEnv(rllib.env.MultiAgentEnv, FactoryEnv):
    """Define a ray multi agent env"""

    def __init__(self, config=None):
        super().__init__(config)
        # All agents active at once. We set this as safeguard to prevent the addition
        # of agent-specific features by accident.
        self.current_agent = None

    def step(self, action: Dict):
        agents = action.keys()

        for agent in agents:
            # Carrying out actions sequentially might lead to certain collisions and invalid rail enterings.
            agent_action = Action(action.get(agent))
            action_result = do_action(
                self.factory.tables[agent], self.factory, agent_action
            )
            self.factory.add_move(agent, agent_action, action_result)

        observations = {i: get_observations(i, self.factory) for i in agents}
        observations = add_masking(self, observations)

        rewards = {i: get_reward(i, self.factory, self.num_episodes) for i in agents}

        # Note: if an agent is "done", we don't get any new actions for said agent
        # in a MultiAgentEnv. This is important, as tables without cores still need
        # to move. We prevent this behaviour by setting all done fields to False until
        # all tables are done.
        all_cores_delivered = all(not t.has_core() for t in self.factory.tables)
        counts = [self.factory.agent_step_counter.get(agent) for agent in agents]
        # maximum steps are counted per agent, not in total (makes it easier to keep config stable)
        max_steps_reached = all(count > self.factory.max_num_steps for count in counts)

        all_done = all_cores_delivered or max_steps_reached

        if all_done:
            dones = {i: True for i in agents}
            dones["__all__"] = True
            self.factory.add_completed_step_count()
            self.num_episodes += 1
            self.factory.print_stats(self.num_episodes)
            self.factory.record_stats()
        else:
            dones = {i: False for i in agents}
            dones["__all__"] = False

        return observations, rewards, dones, {}

    def reset(self):
        if self.config.get("random_init"):
            self.factory = factory_from_config(self.config)
        else:
            self.factory = deepcopy(self.initial_factory)

        self.render()
        observations = {
            i: get_observations(i, self.factory) for i in range(self.num_agents)
        }
        observations = add_masking(self, observations)
        return observations
