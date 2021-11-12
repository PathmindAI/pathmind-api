import gym
import or_gym
import pytest
from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese

from pathmind.simulation import from_gym
from tests.examples.mouse.two_reward_mouse_env_pathmind import TwoRewardMouseAndCheese


def test_training():
    simulation = MouseAndCheese()
    simulation.train()


def test_two_reward_training():
    simulation = TwoRewardMouseAndCheese()
    simulation.train()


def test_multi_training():
    pytest.skip("Needs multi-agent training to work on web app")
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


def test_training_debug_mode():
    # TODO check that training.zip exists
    simulation = MouseAndCheese()
    simulation.train(debug_mode=True)
