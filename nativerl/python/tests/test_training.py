import pytest
import run


@pytest.mark.integration
def test_gym_training():
    run.main(environment="CartPole-v0", is_gym=True, max_episodes=1)


@pytest.mark.integration
def test_or_gym_training():
    run.main(environment="Knapsack-v0", is_gym=True, max_episodes=1)


@pytest.mark.integration
def test_freezing():
    run.main(environment="Knapsack-v0", is_gym=True, max_episodes=1, freezing=True)


@pytest.mark.integration
def test_pathmind_env_module():
    run.main(environment="tests.cartpole.PathmindEnvironment", max_episodes=1)


@pytest.mark.integration
def test_gym_module():
    run.main(environment="tests.gym_cartpole.CartPoleEnv", is_gym=True, max_episodes=1)
