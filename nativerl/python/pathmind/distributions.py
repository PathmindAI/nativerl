import tree
import numpy as np

from ray.rllib.models.tf.tf_action_dist import Categorical, MultiActionDistribution
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType
from ray.rllib.models import ModelCatalog

from ray.rllib.utils.framework import try_import_tf, try_import_tfp

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

# Temperature settings
ICY_TEMP = 0.0000001
COLD_TEMP = 0.01
COOL_TEMP = 0.1
WARM_TEMP = 10.0
HOT_TEMP = 100.0


def register_freezing_distributions(env):

    env_action_space = env.action_space
    if not isinstance(env_action_space, list):  # gym case
        env_action_space = [env_action_space]

    env_output_sizes = [env_action_space[i].n for i in range(len(env_action_space))]
    env_output_shape = sum(env_output_sizes)

    class IcyMultiActionDistribution(MultiActionDistribution):
        def _build_sample_op(self) -> TensorType:
            pass

        @staticmethod
        def required_model_output_shape(action_space, model_config, **kwargs):
            return env_output_shape  # controls model output feature vector size

        def __init__(self, inputs, model, *, child_distributions=[Categorical]*len(env_action_space),
                     input_lens=env_output_sizes,
                     action_space=env_action_space, temperature = ICY_TEMP):
            ActionDistribution.__init__(self, inputs / temperature, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs / temperature, input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

    class IcyCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=ICY_TEMP):
            super().__init__(inputs / temperature, model=None)

    class ColdMultiActionDistribution(MultiActionDistribution):
        def _build_sample_op(self) -> TensorType:
            pass

        @staticmethod
        def required_model_output_shape(action_space, model_config, **kwargs):
            return env_output_shape  # controls model output feature vector size

        def __init__(self, inputs, model, *, child_distributions=[Categorical]*len(env_action_space),
                     input_lens=env_output_sizes,
                     action_space=env_action_space, temperature = COLD_TEMP):
            ActionDistribution.__init__(self, inputs / temperature, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs / temperature, input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

    class ColdCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=COLD_TEMP):
            super().__init__(inputs / temperature, model=None)

    class CoolMultiActionDistribution(MultiActionDistribution):
        def _build_sample_op(self) -> TensorType:
            pass

        @staticmethod
        def required_model_output_shape(action_space, model_config, **kwargs):
            return env_output_shape  # controls model output feature vector size

        def __init__(self, inputs, model, *, child_distributions=[Categorical]*len(env_action_space),
                     input_lens=env_output_sizes,
                     action_space=env_action_space, temperature = COOL_TEMP):
            ActionDistribution.__init__(self, inputs / temperature, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs / temperature, input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

    class CoolCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=COOL_TEMP):
            super().__init__(inputs / temperature, model=None)

    class WarmMultiActionDistribution(MultiActionDistribution):
        def _build_sample_op(self) -> TensorType:
            pass

        @staticmethod
        def required_model_output_shape(action_space, model_config, **kwargs):
            return env_output_shape  # controls model output feature vector size

        def __init__(self, inputs, model, *, child_distributions=[Categorical]*len(env_action_space),
                     input_lens=env_output_sizes,
                     action_space=env_action_space, temperature = WARM_TEMP):
            ActionDistribution.__init__(self, inputs / temperature, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs / temperature, input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

    class WarmCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=WARM_TEMP):
            super().__init__(inputs / temperature, model=None)

    class HotMultiActionDistribution(MultiActionDistribution):
        def _build_sample_op(self) -> TensorType:
            pass

        @staticmethod
        def required_model_output_shape(action_space, model_config, **kwargs):
            return env_output_shape  # controls model output feature vector size

        def __init__(self, inputs, model, *, child_distributions=[Categorical]*len(env_action_space),
                     input_lens=env_output_sizes,
                     action_space=env_action_space, temperature = HOT_TEMP):
            ActionDistribution.__init__(self, inputs / temperature, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs / temperature, input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

    class HotCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=HOT_TEMP):
            super().__init__(inputs / temperature, model=None)

    if len(env_action_space) > 1:
        ModelCatalog.register_custom_action_dist("icy", IcyMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("cold", ColdMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("cool", CoolMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("warm", WarmMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("hot", HotMultiActionDistribution)
    else:
        ModelCatalog.register_custom_action_dist("icy", IcyCategorical)
        ModelCatalog.register_custom_action_dist("cold", ColdCategorical)
        ModelCatalog.register_custom_action_dist("cool", CoolCategorical)
        ModelCatalog.register_custom_action_dist("warm", WarmCategorical)
        ModelCatalog.register_custom_action_dist("hot", HotCategorical)
