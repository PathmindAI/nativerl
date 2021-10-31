import glob
import itertools
import math
import os
import typing
from collections import OrderedDict

import gym
import numpy as np
import yaml
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env

if os.environ.get("USE_PY_NATIVERL"):
    import pathmind_training.pynativerl as nativerl
else:
    import nativerl

import pathmind_training.utils
from pathmind_training.pynativerl import Continuous
from pathmind_training.utils import get_class_from_string

OR_GYM_ENVS = [
    "Knapsack-v0",
    "Knapsack-v1",
    "Knapsack-v2",
    "Knapsack-v3",
    "BinPacking-v0",
    "Newsvendor-v0",
    "VMPacking-v0",
    "VMPacking-v1",
    "VehicleRouting-v0",
    "InvManagement-v0",
    "InvManagement-v1",
    "PortfolioOpt-v0",
    "TSP-v0",
    "TSP-v1",
]


def make_env(env_name):
    if env_name in OR_GYM_ENVS:
        import or_gym

        return or_gym.make(env_name)
    else:
        return gym.make(env_name)


def get_gym_environment(environment_name: str):
    if "." in environment_name:  # a python module, like "cartpole.CartPoleEnv"
        env_class = pathmind_training.utils.get_class_from_string(environment_name)

        def env_creator(env_config):
            return env_class()

        env_name = env_class.__name__

    else:  # built-in gym envs retrieved by name, e.g. "CartPole-v0"
        env_name = environment_name
        try:
            make_env(env_name)
        except Exception:
            raise Exception(
                f"Could not find gym environment '{env_name}'. Make sure to check for typos."
            )

        def env_creator(env_config):
            return make_env(env_name)

    # Register the environment as string
    register_env(env_name, env_creator)

    return env_name, env_creator


def get_environment(
    jar_dir: str,
    environment_name: str,
    is_multi_agent: bool = True,
    max_memory_in_mb: int = 4096,
    is_pathmind_simulation: bool = False,
    obs_selection=None,
):
    base_class = MultiAgentEnv if is_multi_agent else gym.Env

    simple_name = environment_name.split(".")[-1]

    class PathmindEnvironment(base_class):
        def __init__(self, env_config):
            # AnyLogic needs this to find its database
            os.chdir(jar_dir)

            # Put all JAR files found here in the class path
            jars = glob.glob(jar_dir + "/**/*.jar", recursive=True)

            # Initialize nativerl
            nativerl.init(
                [
                    "-Djava.class.path=" + os.pathsep.join(jars + [jar_dir]),
                    f"-Xmx{max_memory_in_mb}m",
                ]
            )

            # Instantiate the native environment, or mock it with pynativerl
            if is_pathmind_simulation:
                self.nativeEnv = get_native_env_from_simulation(
                    environment_name, obs_selection
                )
            else:
                self.nativeEnv = nativerl.createEnvironment(environment_name)

            self.action_space = self.define_action_space()
            self.observation_space = self.define_observation_space()

            self.id = simple_name
            if not is_multi_agent:
                self.unwrapped.spec = self

            self.alphas = env_config["alphas"]
            self.num_reward_terms = len(self.alphas)
            self.betas = np.ones(self.num_reward_terms)

            self.term_contributions_dict = {}
            self.term_contributions = np.zeros(self.num_reward_terms)

        def define_action_space(self):
            i = 0
            action_space = self.nativeEnv.getActionSpace(i)
            action_spaces = []
            while action_space is not None:
                if isinstance(action_space, nativerl.Discrete):
                    action_spaces += [
                        gym.spaces.Discrete(action_space.n)
                        for _ in range(action_space.size)
                    ]
                else:  # Continuous spaces have "shape"
                    action_spaces += [
                        gym.spaces.Box(
                            0, 1, np.array(action_space.shape), dtype=np.float32
                        )
                    ]
                i += 1
                action_space = self.nativeEnv.getActionSpace(i)
            return (
                action_spaces[0]
                if len(action_spaces) == 1
                else gym.spaces.Tuple(action_spaces)
            )

        def define_observation_space(self):
            observation_space = self.nativeEnv.getObservationSpace()
            low = (
                observation_space.low[0]
                if len(observation_space.low) == 1
                else np.array(observation_space.low)
            )
            high = (
                observation_space.high[0]
                if len(observation_space.high) == 1
                else np.array(observation_space.high)
            )
            observation_space = gym.spaces.Box(
                low, high, np.array(observation_space.shape), dtype=np.float32
            )

            action_mask_space = self.nativeEnv.getActionMaskSpace()
            if action_mask_space is not None:
                low = (
                    action_mask_space.low[0]
                    if len(action_mask_space.low) == 1
                    else np.array(action_mask_space.low)
                )
                high = (
                    action_mask_space.high[0]
                    if len(action_mask_space.high) == 1
                    else np.array(action_mask_space.high)
                )
                observation_space = gym.spaces.Dict(
                    {
                        "action_mask": gym.spaces.Box(
                            low,
                            high,
                            np.array(action_mask_space.shape),
                            dtype=np.float32,
                        ),
                        "real_obs": observation_space,
                    }
                )
            return observation_space

        def reset(self):
            self.nativeEnv.reset()

            self.term_contributions_dict = {}
            self.term_contributions = np.zeros(self.num_reward_terms)

            if is_multi_agent:
                obs_dict = {}
                for i in range(0, self.nativeEnv.getNumberOfAgents()):
                    if self.nativeEnv.isSkip(i):
                        continue
                    obs = np.array(self.nativeEnv.getObservation(i))

                    # TODO: this check is weak. Should be checked against an actual action mask parameter
                    if isinstance(self.observation_space, gym.spaces.Dict):
                        obs = {
                            "action_mask": np.array(self.nativeEnv.getActionMask(i)),
                            "real_obs": obs,
                        }
                    obs_dict[str(i)] = obs
                return obs_dict
            else:
                if self.nativeEnv.getNumberOfAgents() != 1:
                    raise ValueError(
                        "Not in multi-agent mode: Number of agents needs to be 1"
                    )
                obs = np.array(self.nativeEnv.getObservation())
                if isinstance(self.observation_space, gym.spaces.Dict):
                    obs = {
                        "action_mask": np.array(self.nativeEnv.getActionMask()),
                        "real_obs": obs,
                    }
                return obs

        def step(self, action):

            if is_multi_agent:
                for i in range(0, self.nativeEnv.getNumberOfAgents()):
                    if self.nativeEnv.isSkip(i):
                        continue
                    if (
                        str(i) in action.keys()
                    ):  # sometimes keys are not present, e.g. when done
                        act = action[str(i)]
                        if isinstance(self.action_space, gym.spaces.Tuple):
                            action_array = np.empty(shape=0, dtype=np.float32)
                            for j in range(0, len(act)):
                                action_array = np.concatenate(
                                    [action_array, act[j].astype(np.float32)], axis=None
                                )
                        else:
                            action_array = act.astype(np.float32)
                        self.nativeEnv.setNextAction(nativerl.Array(action_array), i)

                self.nativeEnv.step()

                obs_dict = {}
                reward_dict = {}
                done_dict = {}
                for i in range(0, self.nativeEnv.getNumberOfAgents()):
                    if self.nativeEnv.isSkip(i):
                        continue
                    obs = np.array(self.nativeEnv.getObservation(i))
                    if isinstance(self.observation_space, gym.spaces.Dict):
                        obs = {
                            "action_mask": np.array(self.nativeEnv.getActionMask(i)),
                            "real_obs": obs,
                        }
                    obs_dict[str(i)] = obs
                    reward_array = np.array(self.nativeEnv.getRewardTerms(i))
                    reward_dict[str(i)] = np.sum(
                        reward_array * self.alphas * self.betas
                    )

                    if str(i) not in self.term_contributions_dict:
                        self.term_contributions_dict[str(i)] = reward_array
                    else:
                        self.term_contributions_dict[str(i)] += reward_array

                    done_dict[str(i)] = self.nativeEnv.isDone(i)

                done_dict["__all__"] = all(done_dict.values())

                if done_dict["__all__"]:
                    self.term_contributions += sum(
                        self.term_contributions_dict.values()
                    ) / len(self.term_contributions_dict)
                return obs_dict, reward_dict, done_dict, {}

            else:
                if self.nativeEnv.getNumberOfAgents() != 1:
                    raise ValueError(
                        "Not in multi-agent mode: Number of agents needs to be 1"
                    )
                if isinstance(self.action_space, gym.spaces.Tuple):
                    action_array = np.empty(shape=0, dtype=np.float32)
                    for j in range(0, len(action)):
                        action_array = np.concatenate(
                            [action_array, action[j].astype(np.float32)], axis=None
                        )
                else:
                    action_array = action.astype(np.float32)
                self.nativeEnv.setNextAction(nativerl.Array(action_array))
                self.nativeEnv.step()

                reward_array = np.array(self.nativeEnv.getRewardTerms())
                reward = np.sum(reward_array * self.alphas * self.betas)
                self.term_contributions += reward_array

                obs = np.array(self.nativeEnv.getObservation())
                done = self.nativeEnv.isDone()

                if isinstance(self.observation_space, gym.spaces.Dict):
                    obs = {
                        "action_mask": np.array(self.nativeEnv.getActionMask()),
                        "real_obs": obs,
                    }
                return obs, reward, done, {}

        def getMetrics(self):
            if is_multi_agent:
                metrics_space = self.nativeEnv.getMetricsSpace()
                # create before the loop to ensure existence in return value
                metrics = np.zeros(metrics_space.shape)
                for i in range(0, self.nativeEnv.getNumberOfAgents()):
                    if self.nativeEnv.isSkip(i):
                        agent_metrics = np.zeros(metrics_space.shape)
                    else:
                        agent_metrics = np.array(self.nativeEnv.getMetrics(i))
                    metrics = (
                        agent_metrics if i == 0 else np.vstack((metrics, agent_metrics))
                    )
                return np.mean(metrics, axis=0)
            else:
                return np.array(self.nativeEnv.getMetrics(0))

        def updateBetas(self, betas):
            self.betas = betas

        def getRewardTermContributions(self):
            return self.term_contributions

    # Set correct class name internally
    PathmindEnvironment.__name__ = simple_name
    PathmindEnvironment.__qualname__ = simple_name

    return PathmindEnvironment


def get_native_env_from_simulation(
    simulation_name: str, observation_file: str = None
) -> nativerl.Environment:
    """Returns a pynativerl.Environment given a Pathmind Python `Simulation` name.

    :param simulation_name: name of the `Simulation` class to run
    :param observation_file: file name of the optional observation yaml
    :return: a `pathmind_training.pynativerl.Environment` instance
    """
    simulation_class = get_class_from_string(simulation_name)

    obs_names: typing.Optional[str] = None
    if observation_file:
        with open(observation_file, "r") as f:
            schema: OrderedDict = yaml.safe_load(f.read())
            obs_names = schema.get("observations")

    class PathmindEnv(nativerl.Environment):
        def __init__(
            self,
            simulation=simulation_class(),
            obs=obs_names,
        ):
            nativerl.Environment.__init__(self)
            self.simulation = simulation
            self.obs_names = obs

        def getActionSpace(self, agent_id=0):
            space = self.simulation.action_space(agent_id=agent_id)
            if hasattr(space, "choices"):  # Discrete space defined
                nativerl_space = nativerl.Discrete(n=space.choices, size=space.size)
            else:  # Continuous space defined
                nativerl_space = nativerl.Continuous(
                    low=[space.low], high=[space.high], shape=space.shape
                )
            return nativerl_space if agent_id < self.getNumberOfAgents() else None

        def getObservationSpace(self):
            obs_shape = [len(self.getObservation(agent_id=0))]
            return nativerl.Continuous([-math.inf], [math.inf], obs_shape)

        def getNumberOfAgents(self):
            return self.simulation.number_of_agents()

        def getActionMask(self, agent_id=0):
            # TODO: support this
            return None

        def getActionMaskSpace(self):
            # TODO: support this
            return None

        def getObservation(self, agent_id=0):
            obs_dict = self.simulation.get_observation(agent_id)

            if not self.obs_names:
                self.obs_names = obs_dict.keys()

            # Flatten all observations here, e.g. [1, 2, [3, 4], 5] => [1, 2, 3, 4, 5]
            lists = [
                [obs_dict[obs]]
                if not isinstance(obs_dict[obs], typing.List)
                else obs_dict[obs]
                for obs in self.obs_names
            ]
            flat_obs = list(itertools.chain(*lists))

            return nativerl.Array(flat_obs)

        def reset(self):
            self.simulation.reset()

        def setNextAction(self, action, agent_id=0):
            if not self.simulation.action:
                self.simulation.action = {}
            self.simulation.action[agent_id] = action

        def isSkip(self, agent_id=0):
            return False

        def step(self):
            return self.simulation.step()

        def isDone(self, agent_id=0):
            return self.simulation.is_done(agent_id)

        def getReward(self, agent_id=0) -> float:
            """In the absence of proper reward term handling for Python models, we just sum up
            all values of the "reward terms" we get from `get_reward`."""
            reward_dict = self.simulation.get_reward(agent_id)
            return sum(reward_dict.values())

        def getMetrics(self, agent_id=0) -> nativerl.Array:
            reward_dict = self.simulation.get_reward(agent_id)
            return nativerl.Array(list(reward_dict.values()))

        def getMetricsSpace(self) -> Continuous:
            num_metrics = len(self.getMetrics())
            return nativerl.Continuous(
                low=[-math.inf], high=[math.inf], shape=[num_metrics]
            )

        def getRewardTerms(self, agent_id: int = 0) -> nativerl.Array:
            reward_dict = self.simulation.get_reward(agent_id)
            return nativerl.Array(reward_dict.values())

    return PathmindEnv()
