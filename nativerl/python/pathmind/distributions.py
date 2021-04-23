import tree
import functools
import numpy as np
import gym

from ray.rllib.models.tf.tf_action_dist import Categorical, DiagGaussian, TFActionDistribution
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, List, Union, \
    Tuple, ModelConfigDict

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
    if isinstance(env.action_space, gym.spaces.Tuple):
        env_output_sizes = [env_action_space[i].n \
                            if env_action_space[i] == gym.spaces.Discrete \
                            else np.prod(env_action_space[i].shape) * 2 \
                            for i in range(len(env_action_space.spaces))]
    else:
        env_output_sizes = [env_action_space.n \
                            if env_action_space == gym.spaces.Discrete \
                            else np.prod(env_action_space.shape) * 2]
    env_output_shape = sum(env_output_sizes)


    class IcyCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=ICY_TEMP):
            super().__init__(inputs / temperature, model=None)

    class IcyDiagGaussian(TFActionDistribution):
        def __init__(self, inputs: List[TensorType], model: ModelV2, temperature=ICY_TEMP):
            mean, log_std = tf.split(inputs, 2, axis=1)
            self.mean = mean
            self.std = tf.exp(log_std) * temperature
            self.log_std = tf.math.log(self.std)
            super().__init__(inputs, model)
    
        @override(ActionDistribution)
        def deterministic_sample(self) -> TensorType:
            return self.mean
    
        @override(ActionDistribution)
        def logp(self, x: TensorType) -> TensorType:
            return -0.5 * tf.reduce_sum(
                tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
                axis=1
            ) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
                tf.reduce_sum(self.log_std, axis=1)
    
        @override(ActionDistribution)
        def kl(self, other: ActionDistribution) -> TensorType:
            assert isinstance(other, IcyDiagGaussian)
            return tf.reduce_sum(
                other.log_std - self.log_std +
                (tf.math.square(self.std) + tf.math.square(self.mean - other.mean))
                / (2.0 * tf.math.square(other.std)) - 0.5,
                axis=1)
    
        @override(ActionDistribution)
        def entropy(self) -> TensorType:
            return tf.reduce_sum(
                self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=1)
    
        @override(TFActionDistribution)
        def _build_sample_op(self) -> TensorType:
            return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
    
        @staticmethod
        @override(ActionDistribution)
        def required_model_output_shape(
                action_space: gym.Space,
                model_config: ModelConfigDict) -> Union[int, np.ndarray]:
            return np.prod(action_space.shape) * 2
    
    class ColdCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=COLD_TEMP):
            super().__init__(inputs / temperature, model=None)

    class ColdDiagGaussian(TFActionDistribution):
        def __init__(self, inputs: List[TensorType], model: ModelV2, temperature=COLD_TEMP):
            mean, log_std = tf.split(inputs, 2, axis=1)
            self.mean = mean
            self.std = tf.exp(log_std) * temperature
            self.log_std = tf.math.log(self.std)
            super().__init__(inputs, model)
    
        @override(ActionDistribution)
        def deterministic_sample(self) -> TensorType:
            return self.mean
    
        @override(ActionDistribution)
        def logp(self, x: TensorType) -> TensorType:
            return -0.5 * tf.reduce_sum(
                tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
                axis=1
            ) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
                tf.reduce_sum(self.log_std, axis=1)
    
        @override(ActionDistribution)
        def kl(self, other: ActionDistribution) -> TensorType:
            assert isinstance(other, ColdDiagGaussian)
            return tf.reduce_sum(
                other.log_std - self.log_std +
                (tf.math.square(self.std) + tf.math.square(self.mean - other.mean))
                / (2.0 * tf.math.square(other.std)) - 0.5,
                axis=1)
    
        @override(ActionDistribution)
        def entropy(self) -> TensorType:
            return tf.reduce_sum(
                self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=1)
    
        @override(TFActionDistribution)
        def _build_sample_op(self) -> TensorType:
            return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
    
        @staticmethod
        @override(ActionDistribution)
        def required_model_output_shape(
                action_space: gym.Space,
                model_config: ModelConfigDict) -> Union[int, np.ndarray]:
            return np.prod(action_space.shape) * 2
    
    class CoolCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=COOL_TEMP):
            super().__init__(inputs / temperature, model=None)

    class CoolDiagGaussian(TFActionDistribution):
        def __init__(self, inputs: List[TensorType], model: ModelV2, temperature=COOL_TEMP):
            mean, log_std = tf.split(inputs, 2, axis=1)
            self.mean = mean
            self.std = tf.exp(log_std) * temperature
            self.log_std = tf.math.log(self.std)
            super().__init__(inputs, model)
    
        @override(ActionDistribution)
        def deterministic_sample(self) -> TensorType:
            return self.mean
    
        @override(ActionDistribution)
        def logp(self, x: TensorType) -> TensorType:
            return -0.5 * tf.reduce_sum(
                tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
                axis=1
            ) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
                tf.reduce_sum(self.log_std, axis=1)
    
        @override(ActionDistribution)
        def kl(self, other: ActionDistribution) -> TensorType:
            assert isinstance(other, CoolDiagGaussian)
            return tf.reduce_sum(
                other.log_std - self.log_std +
                (tf.math.square(self.std) + tf.math.square(self.mean - other.mean))
                / (2.0 * tf.math.square(other.std)) - 0.5,
                axis=1)
    
        @override(ActionDistribution)
        def entropy(self) -> TensorType:
            return tf.reduce_sum(
                self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=1)
    
        @override(TFActionDistribution)
        def _build_sample_op(self) -> TensorType:
            return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
    
        @staticmethod
        @override(ActionDistribution)
        def required_model_output_shape(
                action_space: gym.Space,
                model_config: ModelConfigDict) -> Union[int, np.ndarray]:
            return np.prod(action_space.shape) * 2
    
    class WarmCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=WARM_TEMP):
            super().__init__(inputs / temperature, model=None)

    class WarmDiagGaussian(TFActionDistribution):
        def __init__(self, inputs: List[TensorType], model: ModelV2, temperature=WARM_TEMP):
            mean, log_std = tf.split(inputs, 2, axis=1)
            self.mean = mean
            self.std = tf.exp(log_std) * temperature
            self.log_std = tf.math.log(self.std)
            super().__init__(inputs, model)
    
        @override(ActionDistribution)
        def deterministic_sample(self) -> TensorType:
            return self.mean
    
        @override(ActionDistribution)
        def logp(self, x: TensorType) -> TensorType:
            return -0.5 * tf.reduce_sum(
                tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
                axis=1
            ) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
                tf.reduce_sum(self.log_std, axis=1)
    
        @override(ActionDistribution)
        def kl(self, other: ActionDistribution) -> TensorType:
            assert isinstance(other, WarmDiagGaussian)
            return tf.reduce_sum(
                other.log_std - self.log_std +
                (tf.math.square(self.std) + tf.math.square(self.mean - other.mean))
                / (2.0 * tf.math.square(other.std)) - 0.5,
                axis=1)
    
        @override(ActionDistribution)
        def entropy(self) -> TensorType:
            return tf.reduce_sum(
                self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=1)
    
        @override(TFActionDistribution)
        def _build_sample_op(self) -> TensorType:
            return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
    
        @staticmethod
        @override(ActionDistribution)
        def required_model_output_shape(
                action_space: gym.Space,
                model_config: ModelConfigDict) -> Union[int, np.ndarray]:
            return np.prod(action_space.shape) * 2
    
    class HotCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=HOT_TEMP):
            super().__init__(inputs / temperature, model=None)

    class HotDiagGaussian(TFActionDistribution):
        def __init__(self, inputs: List[TensorType], model: ModelV2, temperature=HOT_TEMP):
            mean, log_std = tf.split(inputs, 2, axis=1)
            self.mean = mean
            self.std = tf.exp(log_std) * temperature
            self.log_std = tf.math.log(self.std)
            super().__init__(inputs, model)
    
        @override(ActionDistribution)
        def deterministic_sample(self) -> TensorType:
            return self.mean
    
        @override(ActionDistribution)
        def logp(self, x: TensorType) -> TensorType:
            return -0.5 * tf.reduce_sum(
                tf.math.square((tf.cast(x, tf.float32) - self.mean) / self.std),
                axis=1
            ) - 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], tf.float32) - \
                tf.reduce_sum(self.log_std, axis=1)
    
        @override(ActionDistribution)
        def kl(self, other: ActionDistribution) -> TensorType:
            assert isinstance(other, HotDiagGaussian)
            return tf.reduce_sum(
                other.log_std - self.log_std +
                (tf.math.square(self.std) + tf.math.square(self.mean - other.mean))
                / (2.0 * tf.math.square(other.std)) - 0.5,
                axis=1)
    
        @override(ActionDistribution)
        def entropy(self) -> TensorType:
            return tf.reduce_sum(
                self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=1)
    
        @override(TFActionDistribution)
        def _build_sample_op(self) -> TensorType:
            return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
    
        @staticmethod
        @override(ActionDistribution)
        def required_model_output_shape(
                action_space: gym.Space,
                model_config: ModelConfigDict) -> Union[int, np.ndarray]:
            return np.prod(action_space.shape) * 2

    if isinstance(env.action_space, gym.spaces.Tuple):
        icy_child_dists = [IcyCategorical \
                           if env_action_space[i] == gym.spaces.Discrete \
                           else IcyDiagGaussian \
                           for i in range(len(env_action_space.spaces))]
        cold_child_dists = [ColdCategorical \
                            if env_action_space[i] == gym.spaces.Discrete \
                            else ColdDiagGaussian \
                            for i in range(len(env_action_space.spaces))]
        cool_child_dists = [CoolCategorical \
                            if env_action_space[i] == gym.spaces.Discrete \
                            else CoolDiagGaussian \
                            for i in range(len(env_action_space.spaces))] 
        warm_child_dists = [WarmCategorical \
                            if env_action_space[i] == gym.spaces.Discrete \
                            else WarmDiagGaussian \
                            for i in range(len(env_action_space.spaces))]
        hot_child_dists = [HotCategorical \
                            if env_action_space[i] == gym.spaces.Discrete \
                            else HotDiagGaussian \
                            for i in range(len(env_action_space.spaces))] 
    else:
        icy_child_dists = [IcyCategorical \
                           if env_action_space == gym.spaces.Discrete \
                           else IcyDiagGaussian]
        cold_child_dists = [ColdCategorical \
                            if env_action_space == gym.spaces.Discrete \
                            else ColdDiagGaussian]
        cool_child_dists = [CoolCategorical \
                            if env_action_space == gym.spaces.Discrete \
                            else CoolDiagGaussian]
        warm_child_dists = [WarmCategorical \
                            if env_action_space == gym.spaces.Discrete \
                            else WarmDiagGaussian]
        hot_child_dists = [HotCategorical \
                           if env_action_space == gym.spaces.Discrete \
                           else HotDiagGaussian]

    class IcyMultiActionDistribution(TFActionDistribution):
        def __init__(self, inputs, model, *, child_distributions=icy_child_dists, input_lens=env_output_sizes,
                     action_space=env_action_space):
            ActionDistribution.__init__(self, inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

        @override(ActionDistribution)
        def logp(self, x):
            # Single tensor input (all merged).
            if isinstance(x, (tf.Tensor, np.ndarray)):
                split_indices = []
                for dist in self.flat_child_distributions:
                    if isinstance(dist, Categorical):
                        split_indices.append(1)
                    else:
                        split_indices.append(tf.shape(dist.sample())[1])
                split_x = tf.split(x, split_indices, axis=1)
            # Structured or flattened (by single action component) input.
            else:
                split_x = tree.flatten(x)

            def map_(val, dist):
                # Remove extra categorical dimension.
                if isinstance(dist, Categorical):
                    val = tf.cast(tf.squeeze(val, axis=-1), tf.int32)
                return dist.logp(val)

            # Remove extra categorical dimension and take the logp of each
            # component.
            flat_logps = tree.map_structure(map_, split_x,
                                            self.flat_child_distributions)

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o) for d, o in zip(self.flat_child_distributions,
                                        other.flat_child_distributions)
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.deterministic_sample(),
                                      child_distributions)

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        #@override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class ColdMultiActionDistribution(TFActionDistribution):
        def __init__(self, inputs, model, *, child_distributions=cold_child_dists, input_lens=env_output_sizes,
                     action_space=env_action_space):
            ActionDistribution.__init__(self, inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

        @override(ActionDistribution)
        def logp(self, x):
            # Single tensor input (all merged).
            if isinstance(x, (tf.Tensor, np.ndarray)):
                split_indices = []
                for dist in self.flat_child_distributions:
                    if isinstance(dist, Categorical):
                        split_indices.append(1)
                    else:
                        split_indices.append(tf.shape(dist.sample())[1])
                split_x = tf.split(x, split_indices, axis=1)
            # Structured or flattened (by single action component) input.
            else:
                split_x = tree.flatten(x)

            def map_(val, dist):
                # Remove extra categorical dimension.
                if isinstance(dist, Categorical):
                    val = tf.cast(tf.squeeze(val, axis=-1), tf.int32)
                return dist.logp(val)

            # Remove extra categorical dimension and take the logp of each
            # component.
            flat_logps = tree.map_structure(map_, split_x,
                                            self.flat_child_distributions)

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o) for d, o in zip(self.flat_child_distributions,
                                        other.flat_child_distributions)
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.deterministic_sample(),
                                      child_distributions)

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        #@override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class CoolMultiActionDistribution(TFActionDistribution):
        def __init__(self, inputs, model, *, child_distributions=cool_child_dists, input_lens=env_output_sizes,
                     action_space=env_action_space):
            ActionDistribution.__init__(self, inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

        @override(ActionDistribution)
        def logp(self, x):
            # Single tensor input (all merged).
            if isinstance(x, (tf.Tensor, np.ndarray)):
                split_indices = []
                for dist in self.flat_child_distributions:
                    if isinstance(dist, Categorical):
                        split_indices.append(1)
                    else:
                        split_indices.append(tf.shape(dist.sample())[1])
                split_x = tf.split(x, split_indices, axis=1)
            # Structured or flattened (by single action component) input.
            else:
                split_x = tree.flatten(x)

            def map_(val, dist):
                # Remove extra categorical dimension.
                if isinstance(dist, Categorical):
                    val = tf.cast(tf.squeeze(val, axis=-1), tf.int32)
                return dist.logp(val)

            # Remove extra categorical dimension and take the logp of each
            # component.
            flat_logps = tree.map_structure(map_, split_x,
                                            self.flat_child_distributions)

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o) for d, o in zip(self.flat_child_distributions,
                                        other.flat_child_distributions)
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.deterministic_sample(),
                                      child_distributions)

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        #@override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class WarmMultiActionDistribution(TFActionDistribution):
        def __init__(self, inputs, model, *, child_distributions=warm_child_dists, input_lens=env_output_sizes,
                     action_space=env_action_space):
            ActionDistribution.__init__(self, inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

        @override(ActionDistribution)
        def logp(self, x):
            # Single tensor input (all merged).
            if isinstance(x, (tf.Tensor, np.ndarray)):
                split_indices = []
                for dist in self.flat_child_distributions:
                    if isinstance(dist, Categorical):
                        split_indices.append(1)
                    else:
                        split_indices.append(tf.shape(dist.sample())[1])
                split_x = tf.split(x, split_indices, axis=1)
            # Structured or flattened (by single action component) input.
            else:
                split_x = tree.flatten(x)

            def map_(val, dist):
                # Remove extra categorical dimension.
                if isinstance(dist, Categorical):
                    val = tf.cast(tf.squeeze(val, axis=-1), tf.int32)
                return dist.logp(val)

            # Remove extra categorical dimension and take the logp of each
            # component.
            flat_logps = tree.map_structure(map_, split_x,
                                            self.flat_child_distributions)

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o) for d, o in zip(self.flat_child_distributions,
                                        other.flat_child_distributions)
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.deterministic_sample(),
                                      child_distributions)

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        #@override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class HotMultiActionDistribution(TFActionDistribution):
        def __init__(self, inputs, model, *, child_distributions=hot_child_dists, input_lens=env_output_sizes,
                     action_space=env_action_space):
            ActionDistribution.__init__(self, inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model), child_distributions,
                split_inputs)

        @override(ActionDistribution)
        def logp(self, x):
            # Single tensor input (all merged).
            if isinstance(x, (tf.Tensor, np.ndarray)):
                split_indices = []
                for dist in self.flat_child_distributions:
                    if isinstance(dist, Categorical):
                        split_indices.append(1)
                    else:
                        split_indices.append(tf.shape(dist.sample())[1])
                split_x = tf.split(x, split_indices, axis=1)
            # Structured or flattened (by single action component) input.
            else:
                split_x = tree.flatten(x)

            def map_(val, dist):
                # Remove extra categorical dimension.
                if isinstance(dist, Categorical):
                    val = tf.cast(tf.squeeze(val, axis=-1), tf.int32)
                return dist.logp(val)

            # Remove extra categorical dimension and take the logp of each
            # component.
            flat_logps = tree.map_structure(map_, split_x,
                                            self.flat_child_distributions)

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o) for d, o in zip(self.flat_child_distributions,
                                        other.flat_child_distributions)
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(self.action_space_struct,
                                                    self.flat_child_distributions)
            return tree.map_structure(lambda s: s.deterministic_sample(),
                                      child_distributions)

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        #@override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    if isinstance(env_action_space, gym.spaces.Discrete):
        ModelCatalog.register_custom_action_dist("icy", IcyCategorical)
        ModelCatalog.register_custom_action_dist("cold", ColdCategorical)
        ModelCatalog.register_custom_action_dist("cool", CoolCategorical)
        ModelCatalog.register_custom_action_dist("warm", WarmCategorical)
        ModelCatalog.register_custom_action_dist("hot", HotCategorical)
    elif isinstance(env_action_space, gym.spaces.Box):
        ModelCatalog.register_custom_action_dist("icy", IcyDiagGaussian)
        ModelCatalog.register_custom_action_dist("cold", ColdDiagGaussian)
        ModelCatalog.register_custom_action_dist("cool", CoolDiagGaussian)
        ModelCatalog.register_custom_action_dist("warm", WarmDiagGaussian)
        ModelCatalog.register_custom_action_dist("hot", HotDiagGaussian)
    elif isinstance(env_action_space, gym.spaces.Tuple):
        ModelCatalog.register_custom_action_dist("icy", IcyMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("cold", ColdMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("cool", CoolMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("warm", WarmMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("hot", HotMultiActionDistribution)
    else:
        print("Custom distributions currently only built for Discrete, Continuous, and Tuple thereof")
