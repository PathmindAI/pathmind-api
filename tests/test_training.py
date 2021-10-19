from pathmind.training import PathmindTraining
from examples.mouse.mouse_env_pathmind import MouseAndCheese


def test_training():

    training = PathmindTraining(simulation=MouseAndCheese())
    training.run()
