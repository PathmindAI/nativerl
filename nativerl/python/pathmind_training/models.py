from ray.rllib.models import MODEL_DEFAULTS, ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf1_module, tf, version = try_import_tf()


def get_custom_model(
    num_hidden_nodes: int,
    num_hidden_layers: int,
    autoregressive: bool,
    action_masking: bool,
    discrete: bool,
):
    model = MODEL_DEFAULTS.copy()

    if action_masking and not discrete:
        msg = "Action masking only supported for discrete actions."
        raise ValueError(msg)

    if action_masking and autoregressive:
        msg = "Action masking and auto-regression can't be enabled simultaneously."
        raise ValueError(msg)

    hidden_layers = [num_hidden_nodes for _ in range(num_hidden_layers)]
    model["fcnet_hiddens"] = hidden_layers
    model["vf_share_layers"] = False

    if autoregressive:
        from pathmind_training.autoregression import (
            get_autoregressive_action_distribution,
            get_autoregressive_actions_model,
        )

        # TODO: need input arguments on run in general
        num_actions = 2
        tuple_length = 3

        model = get_autoregressive_actions_model(
            num_actions=num_actions, tuple_length=tuple_length
        )
        distro = get_autoregressive_action_distribution(tuple_length=tuple_length)

        ModelCatalog.register_custom_model("autoregressive_model", model)
        ModelCatalog.register_custom_action_dist("n_ary_autoreg_output", distro)

        model = {
            "custom_model": "autoregressive_model",
            "custom_action_dist": "nary_autoreg_output",
        }

    if action_masking:
        masking_model = get_action_masking_model(hidden_layers)
        ModelCatalog.register_custom_model("action_masking_tf_model", masking_model)
        model = {"custom_model": "action_masking_tf_model"}

    return model


def get_action_masking_model(hidden_layers):
    class ActionMaskingTFModel(TFModelV2):
        """Custom TF Model that masks out illegal moves. Works for any
        RLlib algorithm (tested only on PPO and DQN so far, though).
        """

        def __init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        ):
            super().__init__(
                obs_space, action_space, num_outputs, model_config, name, **kw
            )

            model_config["fcnet_hiddens"] = hidden_layers
            model_config["vf_share_layers"] = False

            self.base_model = FullyConnectedNetwork(
                obs_space.original_space["real_obs"],
                action_space,
                num_outputs,
                model_config,
                name,
            )

            # Necessary for Ray 1.0.0. Remove for Ray 1.3.0+.
            # self.register_variables(self.base_model.variables())

        def forward(self, input_dict, state, seq_lens):
            logits, _ = self.base_model({"obs": input_dict["obs"]["real_obs"]})
            action_mask = input_dict["obs"]["action_mask"]
            inf_mask = tf.math.maximum(tf.math.log(action_mask), tf.float32.min)
            return logits + inf_mask, state

        def value_function(self):
            return self.base_model.value_function()

        def import_from_h5(self, h5_file):
            pass

    return ActionMaskingTFModel
