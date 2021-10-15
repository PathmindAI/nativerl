from ..config import SIMULATION_CONFIG, MASK_KEY, OBS_KEY
from ..util.samples import factory_from_config
from ..features import get_observations
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.utils.framework import try_import_tf
from gym.spaces import Box

tf = try_import_tf()

MASKING_MODEL_NAME = "action_masking_tf_model"
low = SIMULATION_CONFIG.get("low")
high = SIMULATION_CONFIG.get("high")


def get_num_obs():
    factory = factory_from_config(SIMULATION_CONFIG)
    dummy_obs = get_observations(0, factory)
    del factory
    return len(dummy_obs)


num_obs = get_num_obs()
num_actions = SIMULATION_CONFIG.get("actions")
fcnet_hiddens = SIMULATION_CONFIG.get("fcnet_hiddens")


class ActionMaskingTFModel(DistributionalQTFModel):
    """Custom TF Model that masks out illegal moves. Works for any
    RLlib algorithm (tested only on PPO and DQN so far, though).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kw)

        model_config["fcnet_hiddens"] = fcnet_hiddens

        self.base_model = FullyConnectedNetwork(
            Box(low, high, shape=(num_obs,)),
            action_space,
            num_actions,
            model_config,
            name,
        )

        self.register_variables(self.base_model.variables())

    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.base_model({"obs": input_dict["obs"][OBS_KEY]})
        action_mask = input_dict["obs"][MASK_KEY]
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return logits + inf_mask, state

    def value_function(self):
        return self.base_model.value_function()

    def import_from_h5(self, h5_file):
        pass
