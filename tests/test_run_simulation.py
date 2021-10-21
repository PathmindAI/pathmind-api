import os
import pathlib

import pytest
from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese

from pathmind.policy import Local, Random, Server

PATH = pathlib.Path(__file__).parent.resolve()


def test_single_mouse_rollout():
    simulation = MouseAndCheese()
    policy = Local(model_file=os.path.join(PATH, "examples/mouse_model"))
    simulation.run(policy, out_csv="output.csv")


def test_multi_mouse_rollout():
    simulation = MultiMouseAndCheese()
    policy = Local(model_file=os.path.join(PATH, "examples/multi_mouse_model"))
    simulation.run(policy)


def test_random_single_mouse_rollout():
    simulation = MouseAndCheese()
    policy = Random()
    simulation.run(policy)


def test_random_multi_mouse_rollout():
    simulation = MultiMouseAndCheese()
    policy = Random()
    simulation.run(policy)


def test_server_single_mouse_rollout_locally():
    pytest.skip("Requires running policy server")
    simulation = MouseAndCheese()
    policy = Server(url="localhost:8080", api_key="1234567asdfgh")
    simulation.run(policy)


def test_server_multi_mouse_rollout_locally():
    pytest.skip("Requires running policy server")
    simulation = MultiMouseAndCheese()
    policy = Server(url="localhost:8080", api_key="1234567asdfgh")
    simulation.run(policy)


def test_server_single_mouse_rollout():
    pytest.skip("Requires running policy server")
    simulation = MouseAndCheese()
    policy = Server(url="http://localhost:8000", api_key="1234567asdfgh")
    with pytest.raises(ValueError) as info:
        # This fails due to a validation check
        failed_action = policy.get_actions(simulation)
        assert failed_action is None
    assert "field required" in str(info.value)

    def get_obs(self, agent_id: int):
        return {
            "mouse_row": 1,
            "mouse_col": 1,
            "mouse_row_distance": 1,  # differs from
            "mouse_col_distance": 1,  # current implementation
            "cheese_row": 1,
            "cheese_col": 1,
        }

    # Monkey patch observations to match the expected input of policy server
    old_obs = MouseAndCheese.get_observation
    MouseAndCheese.get_observation = get_obs

    # Get a proper action
    action = policy.get_actions(simulation)
    import numpy

    assert type(action[0]) == numpy.ndarray
    assert 0 <= action[0][0] <= 3

    # Patch back
    MouseAndCheese.get_observation = old_obs
