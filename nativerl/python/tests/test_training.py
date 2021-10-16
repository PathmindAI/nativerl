from random import randint

import pytest
import run


@pytest.mark.integration
def test_gym_training():
    output_dir = f"testoutputs/test-gym-training-{randint(0,1000)}"
    run.main(
        environment="CartPole-v0", is_gym=True, max_episodes=1, output_dir=output_dir
    )


@pytest.mark.integration
def test_or_gym_training():
    output_dir = f"testoutputs/test-or-gym-training-{randint(0,1000)}"
    run.main(
        environment="Knapsack-v0", is_gym=True, max_episodes=1, output_dir=output_dir
    )


@pytest.mark.integration
def test_freezing():
    output_dir = f"testoutputs/test-freezing-{randint(0,1000)}"
    run.main(
        environment="Knapsack-v0",
        is_gym=True,
        max_episodes=1,
        freezing=True,
        output_dir=output_dir,
    )


@pytest.mark.integration
def test_pathmind_env_module():
    output_dir = f"testoutputs/test-pathmind-env-module-{randint(0,1000)}"
    run.main(
        environment="tests.cartpole.PathmindEnvironment",
        max_episodes=1,
        is_pathmind_simulation=True,
        output_dir=output_dir,
    )


@pytest.mark.integration
def test_gym_module():
    output_dir = f"testoutputs/test-gym-module-{randint(0,1000)}"
    run.main(
        environment="tests.gym_cartpole.CartPoleEnv",
        is_gym=True,
        max_episodes=1,
        output_dir=output_dir,
    )
