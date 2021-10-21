import os
import pathlib

import numpy as np
import pandas as pd
import pytest
from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese

from pathmind.policy import Local, Random, Server

PATH = pathlib.Path(__file__).parent.resolve()


def test_single_mouse_rollout():
    simulation = MouseAndCheese()
    policy = Local(model_file=os.path.join(PATH, "examples/mouse_model"))
    simulation.run(policy, num_episodes=10, summary_csv="summary.csv")
    actual = pd.read_csv("summary.csv")
    d = {
        "Episode": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "reward_0_found_cheese": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    expected = pd.DataFrame(data=d)
    assert actual.equals(expected)


def test_multi_mouse_rollout():
    simulation = MultiMouseAndCheese()
    policy = Local(model_file=os.path.join(PATH, "examples/multi_mouse_model"))
    simulation.run(policy, out_csv="output.csv")


def test_random_single_mouse_rollout():
    simulation = MouseAndCheese()
    policy = Random()
    simulation.run(policy)


def test_random_multi_mouse_rollout():
    simulation = MultiMouseAndCheese()
    policy = Random()
    simulation.run(policy)


def test_server_single_mouse_rollout():
    simulation = MouseAndCheese()
    policy = Server(
        url="https://api.dev.devpathmind.com/policy/id7060",
        api_key="a90c01ad-8239-432c-9ebc-c79a79c41a07",
    )
    simulation.run(policy)


@pytest.mark.xfail(reason="Needs a policy server trained for multi-mouse")
def test_server_multi_mouse_rollout():
    simulation = MultiMouseAndCheese()
    policy = Server(
        url="https://api.dev.devpathmind.com/policy/id7060",
        api_key="a90c01ad-8239-432c-9ebc-c79a79c41a07",
    )
    simulation.run(policy)


def test_server_single_mouse_rollout_validation():
    simulation = MouseAndCheese()
    policy = Server(
        url="https://api.dev.devpathmind.com/policy/id7060",
        api_key="a90c01ad-8239-432c-9ebc-c79a79c41a07",
    )

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

    with pytest.raises(ValueError) as info:
        # This fails due to a validation check
        failed_action = policy.get_actions(simulation)
        assert failed_action is None
    assert "field required" in str(info.value)

    # Patch back
    MouseAndCheese.get_observation = old_obs

    # Get a proper action
    action = policy.get_actions(simulation)
    import numpy

    assert type(action[0]) == numpy.ndarray
    assert 0 <= action[0][0] <= 3


def test_policy_predictions():
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
