import os
import pathlib

from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese

from pathmind.policy import Local, Random

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
