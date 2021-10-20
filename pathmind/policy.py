from typing import Dict

import numpy as np
import requests
import tensorflow as tf

from pathmind.simulation import Discrete, Simulation

__all__ = ["Server", "Local", "Random"]


class Policy:
    """A Policy returns actions for each agent in the current state of a Simulation."""

    def get_actions(self, simulation: Simulation) -> Dict[int, np.ndarray]:
        raise NotImplementedError


class Server(Policy):
    """Connect to an existing Pathmind policy server for your simulation."""

    def __init__(self, url, api_key):
        self.url = url + "/predict/"
        self.headers = {"access-token": api_key}

    def get_actions(self, simulation: Simulation) -> Dict[int, np.ndarray]:
        actions = {}
        for i in range(simulation.number_of_agents()):
            obs: dict = simulation.get_observation(i)
            import json

            content = requests.post(
                url=self.url, json=obs, headers=self.headers
            ).content
            actions[i] = np.asarray(json.loads(content).get("actions"))
        return actions


class Local(Policy):
    """Load a policy from a locally stored model file and use it to predict actions."""

    def __init__(self, model_file="./saved_model", is_tuple=False, is_discrete=True):
        self.is_training_tensor = tf.constant(False, dtype=tf.bool)
        self.prev_reward_tensor = tf.constant([0], dtype=tf.float32)
        self.prev_action_tensor = tf.constant([0], dtype=tf.int64)
        self.seq_lens_tensor = tf.constant([0], dtype=tf.int32)
        self.timestep = tf.compat.v1.placeholder_with_default(
            tf.zeros((), dtype=tf.int64), (), name="timestep"
        )

        tf_trackable = tf.saved_model.load(model_file)
        self.model = tf_trackable.signatures.get("serving_default")
        self.is_tuple = is_tuple
        self.is_discrete = is_discrete

    def get_actions(self, simulation: Simulation) -> Dict[int, np.ndarray]:
        """Compute an action by passing observations through a downloaded
        policy_file.zip"""
        action_type = int if self.is_discrete else float

        actions = {}
        for i in range(simulation.number_of_agents()):
            obs: dict = simulation.get_observation(i)
            obs_values: list = [*obs.values()]
            observation = np.asarray(obs_values).reshape((1, -1))
            tensors = tf.convert_to_tensor(
                observation, dtype=tf.float32, name="observations"
            )

            result = self.model(
                observations=tensors,
                is_training=self.is_training_tensor,
                seq_lens=self.seq_lens_tensor,
                prev_action=self.prev_action_tensor,
                prev_reward=self.prev_reward_tensor,
                timestep=self.timestep,
            )

            action_keys = [k for k in result.keys() if "actions_" in k]

            if not self.is_tuple:
                action_tensor = result.get(action_keys[0])
                numpy_tensor = action_tensor.numpy()
                plain_actions = [action_type(numpy_tensor[0])]
            else:
                numpy_tensors = [result.get(k).numpy() for k in action_keys]
                plain_actions = [action_type(x) for x in numpy_tensors]

            actions[i] = np.asarray(plain_actions)

        return actions


class Random(Policy):
    """Generate random actions for a simulaton."""

    def get_actions(self, simulation: Simulation):
        """Generate a random action independent of the observation"""
        actions = {}
        for i in range(simulation.number_of_agents()):
            action_space = simulation.action_space(i)
            if isinstance(action_space, Discrete):
                action = np.random.randint(action_space.choices, size=action_space.size)
            else:
                action = (
                    np.random.rand(*action_space.shape)
                    * (action_space.high - action_space.low)
                    + action_space.low
                )
            actions[i] = action

        return actions
