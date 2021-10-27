import functools

import gym
import numpy as np
import tree
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.tf.tf_action_dist import Categorical, TFActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space

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
        env_output_sizes = [
            env_action_space[i].n for i in range(len(env_action_space.spaces))
        ]
        child_dists = [Categorical] * len(env_action_space.spaces)
    else:
        env_output_sizes = [env_action_space.n]
        child_dists = [Categorical]
    env_output_shape = sum(env_output_sizes)

    class IcyMultiActionDistribution(TFActionDistribution):
        def __init__(
            self,
            inputs,
            model,
            *,
            child_distributions=child_dists,
            input_lens=env_output_sizes,
            action_space=env_action_space,
            temperature=ICY_TEMP
        ):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            ActionDistribution.__init__(self, tempered_inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(tempered_inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model),
                child_distributions,
                split_inputs,
            )

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
            flat_logps = tree.map_structure(
                map_, split_x, self.flat_child_distributions
            )

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o)
                for d, o in zip(
                    self.flat_child_distributions, other.flat_child_distributions
                )
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(
                lambda s: s.deterministic_sample(), child_distributions
            )

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        # @override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class IcyCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=ICY_TEMP):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            super().__init__(tempered_inputs, model=None)

    class ColdMultiActionDistribution(TFActionDistribution):
        def __init__(
            self,
            inputs,
            model,
            *,
            child_distributions=child_dists,
            input_lens=env_output_sizes,
            action_space=env_action_space,
            temperature=COLD_TEMP
        ):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            ActionDistribution.__init__(self, tempered_inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(tempered_inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model),
                child_distributions,
                split_inputs,
            )

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
            flat_logps = tree.map_structure(
                map_, split_x, self.flat_child_distributions
            )

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o)
                for d, o in zip(
                    self.flat_child_distributions, other.flat_child_distributions
                )
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(
                lambda s: s.deterministic_sample(), child_distributions
            )

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        # @override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class ColdCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=COLD_TEMP):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            super().__init__(tempered_inputs, model=None)

    class CoolMultiActionDistribution(TFActionDistribution):
        def __init__(
            self,
            inputs,
            model,
            *,
            child_distributions=child_dists,
            input_lens=env_output_sizes,
            action_space=env_action_space,
            temperature=COOL_TEMP
        ):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            ActionDistribution.__init__(self, tempered_inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(tempered_inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model),
                child_distributions,
                split_inputs,
            )

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
            flat_logps = tree.map_structure(
                map_, split_x, self.flat_child_distributions
            )

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o)
                for d, o in zip(
                    self.flat_child_distributions, other.flat_child_distributions
                )
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(
                lambda s: s.deterministic_sample(), child_distributions
            )

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        # @override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class CoolCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=COOL_TEMP):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            super().__init__(tempered_inputs, model=None)

    class WarmMultiActionDistribution(TFActionDistribution):
        def __init__(
            self,
            inputs,
            model,
            *,
            child_distributions=child_dists,
            input_lens=env_output_sizes,
            action_space=env_action_space,
            temperature=WARM_TEMP
        ):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            ActionDistribution.__init__(self, tempered_inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(tempered_inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model),
                child_distributions,
                split_inputs,
            )

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
            flat_logps = tree.map_structure(
                map_, split_x, self.flat_child_distributions
            )

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o)
                for d, o in zip(
                    self.flat_child_distributions, other.flat_child_distributions
                )
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(
                lambda s: s.deterministic_sample(), child_distributions
            )

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        # @override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class WarmCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=WARM_TEMP):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            super().__init__(tempered_inputs, model=None)

    class HotMultiActionDistribution(TFActionDistribution):
        def __init__(
            self,
            inputs,
            model,
            *,
            child_distributions=child_dists,
            input_lens=env_output_sizes,
            action_space=env_action_space,
            temperature=HOT_TEMP
        ):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            ActionDistribution.__init__(self, tempered_inputs, model)

            self.action_space_struct = get_base_struct_from_space(action_space)

            self.input_lens = np.array(input_lens, dtype=np.int32)
            split_inputs = tf.split(tempered_inputs, self.input_lens, axis=1)
            self.flat_child_distributions = tree.map_structure(
                lambda dist, input_: dist(input_, model),
                child_distributions,
                split_inputs,
            )

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
            flat_logps = tree.map_structure(
                map_, split_x, self.flat_child_distributions
            )

            return functools.reduce(lambda a, b: a + b, flat_logps)

        @override(ActionDistribution)
        def kl(self, other):
            kl_list = [
                d.kl(o)
                for d, o in zip(
                    self.flat_child_distributions, other.flat_child_distributions
                )
            ]
            return functools.reduce(lambda a, b: a + b, kl_list)

        @override(ActionDistribution)
        def entropy(self):
            entropy_list = [d.entropy() for d in self.flat_child_distributions]
            return functools.reduce(lambda a, b: a + b, entropy_list)

        @override(ActionDistribution)
        def sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(lambda s: s.sample(), child_distributions)

        @override(ActionDistribution)
        def deterministic_sample(self):
            child_distributions = tree.unflatten_as(
                self.action_space_struct, self.flat_child_distributions
            )
            return tree.map_structure(
                lambda s: s.deterministic_sample(), child_distributions
            )

        @override(TFActionDistribution)
        def sampled_action_logp(self):
            p = self.flat_child_distributions[0].sampled_action_logp()
            for c in self.flat_child_distributions[1:]:
                p += c.sampled_action_logp()
            return p

        # @override(ActionDistribution)
        @staticmethod
        def required_model_output_shape(action_space, model_config):
            return env_output_shape

    class HotCategorical(Categorical):
        def __init__(self, inputs, model=None, temperature=HOT_TEMP):
            tempered_inputs = tf.math.maximum(inputs / temperature, tf.float32.min)
            super().__init__(tempered_inputs, model=None)

    if isinstance(env_action_space, gym.spaces.Discrete):
        ModelCatalog.register_custom_action_dist("icy", IcyCategorical)
        ModelCatalog.register_custom_action_dist("cold", ColdCategorical)
        ModelCatalog.register_custom_action_dist("cool", CoolCategorical)
        ModelCatalog.register_custom_action_dist("warm", WarmCategorical)
        ModelCatalog.register_custom_action_dist("hot", HotCategorical)
    elif isinstance(env_action_space, gym.spaces.Tuple):
        ModelCatalog.register_custom_action_dist("icy", IcyMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("cold", ColdMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("cool", CoolMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("warm", WarmMultiActionDistribution)
        ModelCatalog.register_custom_action_dist("hot", HotMultiActionDistribution)
    else:
        print("Custom distributions currently only built for Discrete and Tuple")
