from examples.mouse.mouse_env_pathmind import MouseAndCheese
from examples.mouse.multi_mouse_env_pathmind import MultiMouseAndCheese


def test_training():

    simulation = MouseAndCheese()
    simulation.train("./")


# def test_multi_training():
#
#     simulation = MultiMouseAndCheese()
#     simulation.train("./")
