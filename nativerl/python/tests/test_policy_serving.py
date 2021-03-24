from pathmind.simulation import PolicyServer
from tests.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese
import numpy as np


def test_policy_predictions():
    # Note: this requires a policy server to run on localhost 8000 with (any) MouseAndCheese model hosted.
    simulation = MultiMouseAndCheese()
    server = PolicyServer(url="http://localhost:8000", api_key="1234567asdfgh")

    action = server.get_action(simulation)
    assert list(action.keys()) == [0, 1, 2]
    assert type(action[0]) == np.ndarray

    for i in range(10):
        action = server.get_action(simulation)
        simulation.set_action(action)
        simulation.step()

