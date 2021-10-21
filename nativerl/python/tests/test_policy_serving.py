import numpy as np
import pytest
from pathmind.policy import Server
from tests.mouse.mouse_env_pathmind import MouseAndCheese


@pytest.mark.integration
def test_policy_predictions():
    # Note: this requires a policy server to run on localhost 8000 with (any) MouseAndCheese model hosted.
    # server = PolicyServer(url="http://localhost:8000", api_key="1234567asdfgh")
    # Note: this will be hardcoded to a hosted policy server
    server = Server(
        url="https://api.pathmind.com/policy/id17404",
        api_key="6cf587a1-84cc-4cb6-982d-6e0b1e3d45b7",
    )
    simulation = MouseAndCheese()

    action = server.get_actions(simulation)
    assert list(action.keys()) == [0]
    assert type(action[0]) == np.ndarray

    for i in range(10):
        action = server.get_actions(simulation)
        simulation.set_action(action)
        simulation.step()
