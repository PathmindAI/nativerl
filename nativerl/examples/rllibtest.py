import glob, gym, nativerl, numpy, ray, sys, os
from gym.envs.registration import register
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

class MyEnv(gym.Env):
    def __init__(self, env_config):
        # Put all JAR files found here in the class path
        jars = glob.glob("**/*.jar", recursive=True)
        nativerl.init(["-Djava.class.path=" + os.pathsep.join(jars)]);

        self.nativeEnv = nativerl.createEnvironment("traffic_light_opt.TrafficEnvironment")
        actionSpace = self.nativeEnv.getActionSpace()
        observationSpace = self.nativeEnv.getObservationSpace()
        self.action_space = gym.spaces.Discrete(actionSpace.n)
        self.observation_space = gym.spaces.Box(observationSpace.low[0], observationSpace.high[0], numpy.array(observationSpace.shape), dtype=numpy.float32)
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

# Make sure multiple processes can read the database from AnyLogic
with open("database/db.properties", "r+") as f:
    lines = f.readlines()
    if "hsqldb.lock_file=false\n" not in lines:
        f.write("hsqldb.lock_file=false\n")

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 4
trainer = ppo.PPOTrainer(config=config, env=MyEnv)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(100):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

checkpoint = trainer.save()
print("checkpoint saved at", checkpoint)
