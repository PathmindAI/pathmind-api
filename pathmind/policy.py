import json
import requests
from typing import List, Union, Dict, Optional
import numpy as np
import tensorflow as tf
from simulation import Simulation


def server_action(simulation: Simulation) -> Dict[int, Union[float, np.ndarray]]:
    """Connect to an existing policy server and generate an action 
    for your simulation."""
    headers = {'access_token': api_key}
    actions = {}
    for i in range(simulation.number_of_agents()):
        obs: dict = simulation.get_observation(i)
        data = json.dumps(obs)
        content = requests.post(url=url+"/predict/", data=data, headers=headers).content
        actions[i] = np.asarray(json.loads(content).get("actions"))
    return actions

def local_action(simulation, policy_file):
    """Compute an action by passing observations through a downloaded
    policy_file.zip"""
    is_training_tensor = tf.constant(False, dtype=tf.bool)
    prev_reward_tensor = tf.constant([0], dtype=tf.float32)
    prev_action_tensor = tf.constant([0], dtype=tf.int64)
    seq_lens_tensor = tf.constant([0], dtype=tf.int32)

    tf_trackable = tf.saved_model.load('./saved_model')
    self.model = tf_trackable.signatures.get("serving_default")

    observation = np.asarray(self.get_observation()).reshape((1, -1))
    inputs = tf.convert_to_tensor(np.asarray(observation), dtype=tf.float32, name='observations')

    result = self.model(
        observations=inputs,
        is_training=self.is_training_tensor,
        seq_lens=self.seq_lens_tensor,
        prev_action=self.prev_action_tensor,
        prev_reward=self.prev_reward_tensor
    )

    action_keys = [k for k in result.keys() if "actions_" in k]
    action_tensor = result.get(action_keys[0])

    numpy_tensor = action_tensor.numpy()[0]

   ##TODO: is the action necessarily an int?
   #action = int(numpy_tensor[0])
    

def random_action(simulation):
    """Generate a random action independent of the observation"""
    action = np.random.randint(simulation.action_space.choices, size=simulation.action_space.size) 
