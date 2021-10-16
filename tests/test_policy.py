from examples.mouse.mouse_env_pathmind import MouseAndCheese
from pathmind.utils import rollout_episodes
from pathmind.policy import Local


def test_single_mouse():
    env = MouseAndCheese()
    policy = Local()
    rollout_episodes(env, policy)


def test_multi_mouse():
    from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese

    env = MultiMouseAndCheese()
    policy = Local()
    rollout_episodes(env, policy)
