import gym, nativerl, numpy, ray
from gym.envs.registration import register
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.nativeEnv = nativerl.createEnvironment("TrafficEnvironment")
        actionSpace = self.nativeEnv.getActionSpace()
        observationSpace = self.nativeEnv.getObservationSpace()
        self.action_space = gym.spaces.Discrete(actionSpace.n)
        self.observation_space = gym.spaces.Box(numpy.array(observationSpace.low), numpy.array(observationSpace.high), dtype=numpy.float32)
    def reset(self):
        self.nativeEnv.reset()
        return numpy.array(self.nativeEnv.getObservation())
    def step(self, action):
        reward = self.nativeEnv.step(action)
        return numpy.array(self.nativeEnv.getObservation()), reward, self.nativeEnv.isDone(), {}

#register(
#    id='MyEnv-v0',
#    entry_point='__main__:MyEnv',
#)

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
trainer = ppo.PPOTrainer(config=config, env=MyEnv)

for i in range(100):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

