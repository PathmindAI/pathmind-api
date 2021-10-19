from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese

from pathmind.evaluation import rollout_episodes
from pathmind.policy import Local, Random
import pathlib
import os

PATH = pathlib.Path(__file__).parent.resolve()


def test_single_mouse_rollout():
    env = MouseAndCheese()
    policy = Local(model_file=os.path.join(PATH, "examples/mouse_model"))
    rollout_episodes(env, policy, out_csv="output.csv")


def test_multi_mouse_rollout():

    env = MultiMouseAndCheese()
    policy = Local(model_file=os.path.join(PATH, "examples/multi_mouse_model"))
    rollout_episodes(env, policy)


def test_random_single_mouse_rollout():
    env = MouseAndCheese()
    policy = Random()
    rollout_episodes(env, policy)


def test_random_multi_mouse_rollout():

    env = MultiMouseAndCheese()
    policy = Random()
    rollout_episodes(env, policy)
