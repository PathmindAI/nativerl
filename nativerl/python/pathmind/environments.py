import glob
import os

if os.environ.get("USE_PY_NATIVERL"):
    import pathmind.pynativerl as nativerl
else:
    import nativerl

import gym
import numpy as np
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env


OR_GYM_ENVS = ['Knapsack-v0', 'Knapsack-v1', 'Knapsack-v2', 'Knapsack-v3', 'BinPacking-v0',
               'Newsvendor-v0', 'VMPacking-v0', 'VMPacking-v1', 'VehicleRouting-v0', 'InvManagement-v0',
               'InvManagement-v1', 'PortfolioOpt-v0', 'TSP-v0', 'TSP-v1']


def make_env(env_name):
    if env_name in OR_GYM_ENVS:
        import or_gym
        return or_gym.make(env_name)
    else:
        return gym.make(env_name)


def get_gym_environment(environment_name: str):
    if "." in environment_name:  # a python module, like "cartpole.CartPoleEnv"
        env_class = nativerl.get_environment_class(environment_name)

        def env_creator(env_config):
            return env_class()

        env_name = env_class.__name__

    else:  # built-in gym envs retrieved by name, e.g. "CartPole-v0"
        env_name = environment_name
        try:
            make_env(env_name)
        except Exception:
            raise Exception(f"Could not find gym environment '{env_name}'. Make sure to check for typos.")

        def env_creator(env_config):
            return make_env(env_name)

    # Register the environment as string
    register_env(env_name, env_creator)

    return env_name, env_creator


def get_environment(jar_dir: str, environment_name: str, is_multi_agent: bool = True, max_memory_in_mb: int = 4096):
    simple_name = environment_name.split(".")[-1]
    base_class = MultiAgentEnv if is_multi_agent else gym.Env

    class PathmindEnvironment(base_class):

        def __init__(self, env_config):
            # AnyLogic needs this to find its database
            os.chdir(jar_dir)

            # Put all JAR files found here in the class path
            jars = glob.glob(jar_dir + '/**/*.jar', recursive=True)

            # Initialize nativerl
            nativerl.init(['-Djava.class.path=' + os.pathsep.join(jars + [jar_dir]), f'-Xmx{max_memory_in_mb}m'])

            # Instantiate the native environment, or mock it with pynativerl
            self.nativeEnv = nativerl.createEnvironment(environment_name)

            self.action_space = self.define_action_space()
            self.observation_space = self.define_observation_space()

            self.id = simple_name
            if not is_multi_agent:
                self.unwrapped.spec = self

            self.num_reward_terms = 1

        def define_action_space(self):
            i = 0
            action_space = self.nativeEnv.getActionSpace(i)
            action_spaces = []
            while action_space is not None:
                if isinstance(action_space, nativerl.Discrete):
                    action_spaces += [gym.spaces.Discrete(action_space.n) for _ in range(action_space.size)]
                else:  # Continuous spaces have "shape"
                    action_spaces += [gym.spaces.Box(0, 1, np.array(action_space.shape), dtype=np.float32)]
                i += 1
                action_space = self.nativeEnv.getActionSpace(i)
            return action_spaces[0] if len(action_spaces) == 1 else gym.spaces.Tuple(action_spaces)

        def define_observation_space(self):
            observation_space = self.nativeEnv.getObservationSpace()
            low = observation_space.low[0] if len(observation_space.low) == 1 else np.array(observation_space.low)
            high = observation_space.high[0] if len(observation_space.high) == 1 else np.array(observation_space.high)
            observation_space = gym.spaces.Box(low, high, np.array(observation_space.shape), dtype=np.float32)

            action_mask_space = self.nativeEnv.getActionMaskSpace()
            if action_mask_space is not None:
                low = action_mask_space.low[0] if len(action_mask_space.low) == 1 else np.array(action_mask_space.low)
                high = action_mask_space.high[0] if len(action_mask_space.high) == 1 \
                    else np.array(action_mask_space.high)
                observation_space = gym.spaces.Dict({
                    "action_mask": gym.spaces.Box(low, high, np.array(action_mask_space.shape), dtype=np.float32),
                    "real_obs": observation_space
                })
            return observation_space

        def reset(self):
            self.nativeEnv.reset()

            if is_multi_agent:
                obs_dict = {}
                for i in range(0, self.nativeEnv.getNumberOfAgents()):
                    if self.nativeEnv.isSkip(i):
                        continue
                    obs = np.array(self.nativeEnv.getObservation(i))

                    # TODO: this check is weak. Should be checked against an actual action mask parameter
                    if isinstance(self.observation_space, gym.spaces.Dict):
                        obs = {"action_mask": np.array(self.nativeEnv.getActionMask(i)), "real_obs": obs}
                    obs_dict[str(i)] = obs
                return obs_dict
            else:
                if self.nativeEnv.getNumberOfAgents() != 1:
                    raise ValueError("Not in multi-agent mode: Number of agents needs to be 1")
                obs = np.array(self.nativeEnv.getObservation())
                if isinstance(self.observation_space, gym.spaces.Dict):
                    obs = {"action_mask": np.array(self.nativeEnv.getActionMask()), "real_obs": obs}
                return obs

        def step(self, action):

            if is_multi_agent:
                for i in range(0, self.nativeEnv.getNumberOfAgents()):
                    if self.nativeEnv.isSkip(i):
                        continue
                    act = action[str(i)]
                    if isinstance(self.action_space, gym.spaces.Tuple):
                        action_array = np.empty(shape=0, dtype=np.float32)
                        for j in range(0, len(act)):
                            action_array = np.concatenate([action_array, act[j].astype(np.float32)], axis=None)
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
                            "real_obs": obs
                        }
                    obs_dict[str(i)] = obs
                    reward_dict[str(i)] = self.nativeEnv.getReward(i)
                    done_dict[str(i)] = self.nativeEnv.isDone(i)

                # TODO: why is "all" true, if the last agent is done? should check for all(done_dict)...
                done_dict['__all__'] = self.nativeEnv.isDone(-1)
                return obs_dict, reward_dict, done_dict, {}

            else:
                if self.nativeEnv.getNumberOfAgents() != 1:
                    raise ValueError("Not in multi-agent mode: Number of agents needs to be 1")
                if isinstance(self.action_space, gym.spaces.Tuple):
                    action_array = np.empty(shape=0, dtype=np.float32)
                    for j in range(0, len(action)):
                        action_array = np.concatenate([action_array, action[j].astype(np.float32)], axis=None)
                else:
                    action_array = action.astype(np.float32)
                self.nativeEnv.setNextAction(nativerl.Array(action_array))
                self.nativeEnv.step()
                reward = self.nativeEnv.getReward()
                obs = np.array(self.nativeEnv.getObservation())
                done = self.nativeEnv.isDone()

                if isinstance(self.observation_space, gym.spaces.Dict):
                    obs = {"action_mask": np.array(self.nativeEnv.getActionMask()), "real_obs": obs}
                return obs, reward, done, {}

        def getMetrics(self):
            if is_multi_agent:
                metrics = np.empty(shape=0, dtype=np.float32)
                metrics_space = self.nativeEnv.getMetricsSpace()
                for i in range(0, self.nativeEnv.getNumberOfAgents()):
                    if self.nativeEnv.isSkip(i):
                        metrics = np.concatenate([metrics, np.zeros(metrics_space.shape)], axis=None)
                    else:
                        metrics = np.concatenate([metrics, np.array(self.nativeEnv.getMetrics(i))], axis=None)
                return metrics
            else:
                return np.array(self.nativeEnv.getMetrics(0))

        def updateReward(self, betas):
            self.nativeEnv.updateReward(betas=betas)

    # Set correct class name internally
    PathmindEnvironment.__name__ = simple_name
    PathmindEnvironment.__qualname__ = simple_name

    return PathmindEnvironment
