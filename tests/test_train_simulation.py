import gym
import or_gym
import pytest
from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese

from pathmind.simulation import from_gym


def test_training():
    simulation = MouseAndCheese()
    simulation.train()


@pytest.mark.xfail(reason="Multi-agent hasn't been released to prod yet")
def test_multi_training():
    simulation = MultiMouseAndCheese()
    simulation.train()


def test_from_gym():
    env = gym.make("CartPole-v0")
    sim = from_gym(env)
    sim.train()


def test_from_or_gym():
    env = or_gym.make("Knapsack-v0")
    sim = from_gym(env)
    sim.train()
