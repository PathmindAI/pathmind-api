import pytest
from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese


def test_training():
    simulation = MouseAndCheese()
    simulation.train()


def test_multi_training():
    pytest.skip("Needs multi-agent training to work on web app")
    simulation = MultiMouseAndCheese()
    simulation.train()
