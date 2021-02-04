import gym
from gym.spaces import Discrete, Tuple
import argparse
import random

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf1_module, tf, version = try_import_tf()


def get_distribution(distribution, *args):
    if not args:
        batch = tf.shape(distribution.inputs)[0]
        logits = distribution.model.action_model(
            [distribution.inputs, tf.zeros((batch, 1)), tf.zeros((batch, 1))])
        dist = Categorical(logits[0])
    else:
        vectors = []
        for arg in args:
            # TODO how to do this properly in all cases?
            # vectors.append(tf.expand_dims(tf.cast(arg, tf.float32), 1)) # when a2_context is only a1
            vectors.append(tf.cast(arg, tf.float32))  # when a2_context is a1 & context vector (from obs)
        logits = distribution.model.action_model([distribution.inputs, *vectors])
        dist = Categorical(logits[len(args) - 1])
    return dist


def get_autoregressive_action_distribution(tuple_length: int, model_output_shape: int = 32):
    class AutoRegressiveActionDistribution(ActionDistribution):
        """Action distribution P(a1, a2, ..., aN) = P(a1) * P(a2 | a1) * ... * P(aN | a1, a2, ..., aN-1)"""

        @staticmethod
        def required_model_output_shape(self, model_config):
            return model_output_shape  # controls model output feature vector size, choice

        def deterministic_sample(self):
            log_prob_sum = 0
            samples = []
            for i in range(tuple_length):
                dist = get_distribution(self, *samples)
                sample = dist.deterministic_sample()
                log_prob_sum += dist.logp(sample)
                samples.append(sample)

            self._action_logp = log_prob_sum

            return tuple(samples)

        def sample(self):
            log_prob_sum = 0
            samples = []
            for i in range(tuple_length):
                dist = get_distribution(self, *samples)
                sample = dist.sample()
                log_prob_sum += dist.logp(sample)
                samples.append(sample)

            self._action_logp = log_prob_sum

            return tuple(samples)

        def logp(self, actions):
            sliced_actions = [actions[:, i] for i in range(tuple_length)]
            vectors = [tf.expand_dims(tf.cast(sliced_actions[i], tf.float32), 1) for i in range(tuple_length - 1)]
            logits = self.model.action_model([self.inputs, *vectors])
            log_probs = [Categorical(logits[i]).logp(sliced_actions[i]) for i in range(tuple_length - 1)]

            return tuple(log_probs)

        def sampled_action_logp(self):
            return tf.exp(self._action_logp)

        def entropy(self):
            entropy_sum = 0
            samples = []
            for i in range(tuple_length):
                dist = get_distribution(self, samples)
                samples.append(dist.sample())
                entropy_sum += dist.entropy()

            return entropy_sum

        def kl(self, other):
            terms = 0
            samples = []
            for i in range(tuple_length):
                dist = get_distribution(self, samples)
                samples.append(dist.sample())
                terms += dist.kl(get_distribution(other))

            return terms

    return AutoRegressiveActionDistribution


def get_autoregressive_actions_model(num_actions: int, tuple_length: int):

    class AutoregressiveActionsModel(TFModelV2):
        """Implements the `.action_model` branch required above."""

        def __init__(self, obs_space, action_space, num_outputs, model_config,
                     name):
            super(AutoregressiveActionsModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name)
            if action_space != Tuple([Discrete(num_actions) for _ in range(tuple_length)]):
                raise ValueError(
                    f"This model only supports the {[[num_actions for _ in range(tuple_length)]]} action space")

            # Inputs
            obs_input = tf.keras.layers.Input(
                shape=obs_space.shape, name="obs_input")

            # action_inputs = [tf.keras.layers.Input(shape=(1, ), name=f"a{i+1}_input") for i in range(tuple_length - 1)]
            a1_input = tf.keras.layers.Input(shape=(1, ), name="a1_input")
            a2_input = tf.keras.layers.Input(shape=(1, ), name="a2_input")
            # a3_input = tf.keras.layers.Input(shape=(1, ), name="a3_input")
            action_inputs = [a1_input, a2_input]#, a3_input]

            ctx_input = tf.keras.layers.Input(
                shape=(num_outputs, ), name="ctx_input")

            # Output of the model (normally 'logits', but for an autoregressive
            # dist this is more like a context/feature layer encoding the obs)
            context = tf.keras.layers.Dense(
                num_outputs,
                name="hidden",
                activation=tf.nn.tanh,
                kernel_initializer=normc_initializer(1.0))(obs_input)

            # V(s)
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(context)

            # P(a1 | obs)
            a1_logits = tf.keras.layers.Dense(
                2,
                name="a1_logits",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(ctx_input)

            # P(a2 | a1)
            # --note: typically you'd want to implement P(a2 | a1, obs) as follows:
            a2_context = tf.keras.layers.Concatenate(axis=1)(
                [ctx_input, action_inputs[0]])
            # a2_context = action_inputs[0]
            a2_hidden = tf.keras.layers.Dense(
                64, # hyper-parameter
                name="a2_hidden",
                activation=tf.nn.tanh,
                kernel_initializer=normc_initializer(1.0))(a2_context)
            a2_logits = tf.keras.layers.Dense(
                num_actions,
                name="a2_logits",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(a2_hidden)

            logits = [a1_logits, a2_logits]
            # contexts = [None, a2_context]

            # P(a3 | a1, a2)
            # a3_context = tf.keras.layers.Concatenate(axis=1)(
            #       [a2_context, action_inputs[1]])

            a3_context = tf.keras.layers.Concatenate(axis=1)(
                [ctx_input, action_inputs[0], action_inputs[1]])

            # contexts.append(a3_context)

            a3_hidden = tf.keras.layers.Dense(
                64,  # hyperparameter choice
                name="a3_hidden",
                activation=tf.nn.tanh,
                kernel_initializer=normc_initializer(1.0))(a3_context)
            a3_logits = tf.keras.layers.Dense(
                num_actions,  # number of possible choices for action 3
                name="a3_logits",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(a3_hidden)

            logits.append(a3_logits)

            # Base layers
            self.base_model = tf.keras.Model(obs_input, [context, value_out])
            self.register_variables(self.base_model.variables)
            self.base_model.summary()

            # Autoregressive action sampler

            self.action_model = tf.keras.Model([ctx_input, *action_inputs], logits)
            self.action_model.summary()
            self.register_variables(self.action_model.variables)

        def forward(self, input_dict, state, seq_lens):
            context, self._value_out = self.base_model(input_dict["obs"])
            return context, state

        def value_function(self):
            return tf.reshape(self._value_out, [-1])

        def import_from_h5(self, h5_file: str) -> None:
            pass

    return AutoregressiveActionsModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="PPO")
    parser.add_argument("--stop", type=int, default=200)
    parser.add_argument("--num-cpus", type=int, default=0)
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    num_actions = 2
    tuple_length = 3

    model = get_autoregressive_actions_model(num_actions=num_actions, tuple_length=tuple_length)
    distro = get_autoregressive_action_distribution(tuple_length=tuple_length)

    ModelCatalog.register_custom_model("autoregressive_model", model)
    ModelCatalog.register_custom_action_dist("n_ary_autoreg_output", distro)

    # Test environment
    class CorrelatedActionsEnv(gym.Env):
        """Simple env in which the policy has to emit a tuple of equal actions.
        The best score would be ~200 reward."""

        def __init__(self, _):
            self.observation_space = Discrete(num_actions)
            self.action_space = Tuple([Discrete(num_actions) for _ in range(tuple_length)])

        def reset(self):
            self.t = 0
            self.last = random.choice([i for i in range(num_actions)])
            return self.last

        def step(self, action):
            self.t += 1
            reward = 0

            last_action = self.last
            for a in list(action):
                if a == last_action:
                    reward += 5
                last_action = a

            done = self.t > 20
            self.last = random.choice([i for i in range(num_actions)])
            return self.last, reward, done, {}

        def render(self, mode='human'):
            pass

    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config={
            "env": CorrelatedActionsEnv,
            "gamma": 0.5,
            "num_gpus": 0,
            "model": {
                "custom_model": "autoregressive_model",
                "custom_action_dist": "n_ary_autoreg_output",
            },
        })

