import pytest
from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese


def test_training():
    simulation = MouseAndCheese()
    simulation.train()

@pytest.mark.xfail(reason="Multi-agent hasn't been released to prod yet")
def test_multi_training():
    simulation = MultiMouseAndCheese()
    simulation.train()
