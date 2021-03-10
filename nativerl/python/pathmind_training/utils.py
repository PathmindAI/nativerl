import os
import gym
import math
import numpy as np
if os.environ.get("USE_PY_NATIVERL"):
    import pathmind_training.pynativerl as nativerl
else:
    import nativerl


def write_file(messages, file_name, output_dir, algorithm, mode="a"):
    with open(f"{output_dir}/{algorithm}/{file_name}", mode) as out:
        for msg in messages:
            out.write(f"{msg}\n")


def write_completion_report(trials, output_dir, algorithm):
    # Write to file for Pathmind webapp
    best_trial = "Best Trial: " + str(trials.get_best_trial(metric="episode_reward_mean", mode="max"))
    max_log_dir = trials.get_best_logdir(metric="episode_reward_mean", mode="max")
    best_trial_dir = f"Best Trial Directory: {max_log_dir}"
    best_policy = f"Best Policy: {max_log_dir}/model"
    max_checkpoint = trials.get_best_checkpoint(trial=max_log_dir, metric="episode_reward_mean", mode="max")
    best_checkpoint = f"Best Checkpoint: {max_checkpoint}"

    write_file([best_trial, best_trial_dir, best_policy, best_checkpoint], "ExperimentCompletionReport.txt",
               output_dir, algorithm)

    if trials:
        write_file(["Success: Training completed successfully"], "ExperimentCompletionReport.txt",
                   output_dir, algorithm)
        print("Training completed successfully")


def modify_anylogic_db_properties():
    # Make sure multiple processes can read the database from AnyLogic
    db_props = 'database/db.properties'
    db_lock_line = 'hsqldb.lock_file=false\n'
    if os.path.isfile(db_props):
        with open(db_props, 'r+') as f:
            lines = f.readlines()
            if db_lock_line not in lines:
                f.write(db_lock_line)


def get_mock_env(env):
    class MockEnvironment(gym.Env):
        """Dummy environment with action and observation space matching that of an actual environment,
        used for re-instantiating an agent."""

        def __init__(self, env_config):
            self.action_space = env.action_space
            self.observation_space = env.observation_space

        def reset(self):
            obs = np.array([0])
            return obs

        def step(self, action):
            obs = np.array([0])
            reward = 0
            done = False
            return obs, reward, done, {}

        def render(self, mode='human'):
            pass

    return MockEnvironment


def get_py_nativerl_from_gym_env(env_name: str):
    """Creates a Pathmind Environment from a simple gym.Env with discrete actions and 'Box' observations."""

    env: gym.Env = nativerl.createEnvironment(env_name)
    assert isinstance(env, gym.Env), f"Provided environment {env_name} is not a gym environment."

    assert isinstance(env.action_space, gym.spaces.Discrete), "Only works with discrete actions"
    assert isinstance(env.observation_space, gym.spaces.Box), "Only works with simple 'Box' observations."

    n = env.action_space.n
    num_obs = len(env.observation_space.sample())

    class PathmindEnv(nativerl.Environment):

        def __init__(self, env_config):
            nativerl.Environment.__init__(self)
            self.env = env
            self.obs = []
            self.reward = None
            self.done = False
            self.action = None

        def reset(self):
            self.env.reset()

        def setNextAction(self, action, agent_id=0):
            self.env.action = action.values()[0]

        def step(self):
            obs, self.reward, self.done, _ = self.env.step(self.action)
            self.obs = list(obs)

        def getObservation(self, agent_id=0):
            return np.array(self.obs)

        def isDone(self, agent_id=0):
            return self.done

        def getReward(self, agent_id=0):
            return self.reward

        def getActionSpace(self, agent_id: int = 0):
            return nativerl.Discrete(n) if agent_id == 0 else None

        def getObservationSpace(self):
            return nativerl.Continuous([-math.inf], [math.inf], [num_obs])

        def getActionMaskSpace(self):
            return None

        def getNumberOfAgents(self):
            return 1

        def getActionMask(self, agent_id=0):
            return None

        def isSkip(self, agent_id=0):
            return False

        def getMetrics(self, agent_id=0):
            return np.array([])

        def getMetricsSpace(self) -> nativerl.Continuous:
            pass

    return PathmindEnv
