import pytest
import os

from pathmind_training.environments import make_env, get_environment, get_gym_environment
from pathmind_training.callbacks import get_callbacks
from pathmind_training.distributions import register_freezing_distributions
from pathmind_training.loggers import get_loggers
from pathmind_training.models import get_action_masking_model, get_custom_model
from pathmind_training.pynativerl import get_environment_class, createEnvironment
from pathmind_training.scheduler import get_scheduler
from pathmind_training.stopper import Stopper


def test_make_envs():
    make_env("CartPole-v0")
    make_env("Knapsack-v0")


def test_get_gym_env():
    get_gym_environment("CartPole-v0")
    get_gym_environment("Knapsack-v0")


def test_get_pathmind_env():
    jar_dir = os.getcwd()
    os.chdir(jar_dir)

    get_environment(environment_name="tests.cartpole.PathmindEnvironment", jar_dir=jar_dir)


def test_callbacks():
    get_callbacks(debug_metrics=False, is_gym=True)
    get_callbacks(debug_metrics=True, is_gym=False)
    get_callbacks(debug_metrics=False, is_gym=False)
    get_callbacks(debug_metrics=True, is_gym=True)


def test_distributions():
    env = make_env("CartPole-v0")
    register_freezing_distributions(env=env)


def test_loggers():
    get_loggers()


def test_models():
    get_action_masking_model([256, 256])

    get_custom_model(num_hidden_nodes=2, num_hidden_layers=2,
                     autoregressive=False, action_masking=False, discrete=True)


def test_pynativerl():
    env = createEnvironment("tests.cartpole.PathmindEnvironment")
    assert hasattr(env, "getMetrics")

    clz = get_environment_class("tests.cartpole.PathmindEnvironment")
    assert env.__class__ == clz


def test_scheduler():
    get_scheduler()


def test_stopper():
    stopper = Stopper(output_dir=".", algorithm="PPO", max_iterations=1, max_time_in_sec=10, max_episodes=1,
                      episode_reward_range_th=0.1, entropy_slope_th=0.1, vf_loss_range_th=0.1, value_pred_th=0.1)
    assert hasattr(stopper, "stop")
