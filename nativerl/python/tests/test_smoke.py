import os

import pytest
from pathmind_training.callbacks import get_callbacks
from pathmind_training.distributions import register_freezing_distributions
from pathmind_training.environments import (
    get_environment,
    get_gym_environment,
    make_env,
)
from pathmind_training.loggers import get_loggers
from pathmind_training.models import get_action_masking_model, get_custom_model
from pathmind_training.scheduler import get_scheduler
from pathmind_training.stopper import Stopper
from pathmind_training.utils import createEnvironment, get_class_from_string
from run import test as ma_test


def test_make_envs():
    make_env("CartPole-v0")
    make_env("Knapsack-v0")


def test_get_gym_env():
    get_gym_environment("CartPole-v0")
    get_gym_environment("Knapsack-v0")


def test_get_pathmind_env():
    jar_dir = os.getcwd()
    os.chdir(jar_dir)

    get_environment(
        environment_name="tests.cartpole.PathmindEnvironment", jar_dir=jar_dir
    )


def test_callbacks():
    get_callbacks(
        debug_metrics=False, is_gym=True, use_reward_terms=False, checkpoint_frequency=1
    )
    get_callbacks(
        debug_metrics=True, is_gym=False, use_reward_terms=False, checkpoint_frequency=1
    )
    get_callbacks(
        debug_metrics=False,
        is_gym=False,
        use_reward_terms=False,
        checkpoint_frequency=1,
    )
    get_callbacks(
        debug_metrics=True, is_gym=True, use_reward_terms=False, checkpoint_frequency=1
    )
    get_callbacks(
        debug_metrics=False, is_gym=True, use_reward_terms=True, checkpoint_frequency=1
    )
    get_callbacks(
        debug_metrics=True, is_gym=False, use_reward_terms=True, checkpoint_frequency=1
    )
    get_callbacks(
        debug_metrics=False, is_gym=False, use_reward_terms=True, checkpoint_frequency=1
    )
    get_callbacks(
        debug_metrics=True, is_gym=True, use_reward_terms=True, checkpoint_frequency=1
    )


def test_distributions():
    env = make_env("CartPole-v0")
    register_freezing_distributions(env=env)


def test_loggers():
    get_loggers()


def test_models():
    get_action_masking_model([256, 256])

    get_custom_model(
        num_hidden_nodes=2,
        num_hidden_layers=2,
        autoregressive=False,
        action_masking=False,
        discrete=True,
    )


def test_pynativerl():
    env = createEnvironment("tests.cartpole.PathmindEnvironment")
    assert hasattr(env, "getMetrics")

    clz = get_class_from_string("tests.cartpole.PathmindEnvironment")
    assert env.__class__ == clz


def test_scheduler():
    get_scheduler("PBT")
    get_scheduler("PB2")


def test_stopper():
    stopper = Stopper(
        output_dir=".",
        algorithm="PPO",
        max_iterations=1,
        max_time_in_sec=10,
        max_episodes=1,
        episode_reward_range_th=0.1,
        entropy_slope_th=0.1,
        vf_loss_range_th=0.1,
        value_pred_th=0.1,
        convergence_check_start_iteration=1,
    )
    assert hasattr(stopper, "stop")


def test_ma_test_pathmind(capfd):
    original_dir = os.path.dirname(__file__)
    test_dir = f"{original_dir}/mouse"
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        ma_test(environment="mouse_env_pathmind.MouseAndCheese", module_path=test_dir)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0
    out, err = capfd.readouterr()
    assert "model-analyzer-mode:pm_single" in out


def test_ma_test_pathmind_bad_env(capfd):
    original_dir = os.path.dirname(__file__)
    test_dir = f"{original_dir}/mouse"
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        ma_test(environment="mouse_env_pathmind1.MouseAndCheese", module_path=test_dir)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == -1
    out, err = capfd.readouterr()
    assert "model-analyzer-error:No module named 'mouse_env_pathmind1'" in out


def test_ma_test_gym_env(capfd):
    test_dir = os.path.dirname(__file__)
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        ma_test(environment="gym_cartpole.CartPoleEnv", module_path=test_dir)
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0
    out, err = capfd.readouterr()
    assert "model-analyzer-mode:py_single" in out
