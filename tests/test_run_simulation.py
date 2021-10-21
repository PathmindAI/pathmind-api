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
    simulation.run(policy, num_episodes=10, summary_csv="summary.csv")


def test_random_single_mouse_rollout():
    simulation = MouseAndCheese()
    policy = Random()
    simulation.run(policy)


def test_random_multi_mouse_rollout():
    simulation = MultiMouseAndCheese()
    policy = Random()
    simulation.run(policy)


def test_server_single_mouse_rollout():
    pytest.skip("Requires running policy server")
    simulation = MouseAndCheese()
    policy = Server(url="localhost:8080", api_key="1234567asdfgh")
    simulation.run(policy)


def test_server_multi_mouse_rollout():
    pytest.skip("Requires running policy server")
    simulation = MultiMouseAndCheese()
    policy = Server(url="localhost:8080", api_key="1234567asdfgh")
    simulation.run(policy)
