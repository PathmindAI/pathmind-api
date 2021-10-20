import csv
import math
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
from prettytable import PrettyTable

__all__ = ["Discrete", "Continuous", "Simulation"]


class Discrete:
    """A discrete action space of given size, with the specified number of choices.

    For instance, a Discrete(2) corresponds to a binary choice (0 or 1),
    a Discrete(10) corresponds to an action space with 10 discrete options (0 to 9)
    and a Discrete(3, 2) represents vectors of length two, each with 3 choices, so
    a valid choice would be [0, 1] or [2, 2].
    """

    def __init__(self, choices: int, size: int = 1):
        self.choices = choices
        self.size = size


class Continuous:
    """An action space with continuous values of given shape with specified
    value ranges between "low" and "high".

    For instance, a Continuous([3], 0, 1) has length 3 vectors with values in
    the interval [0, 1] each, whereas a Continuous([3, 2]) accepts values of
    shape (3,2).
    """

    def __init__(
        self, shape: List[int], low: float = -math.inf, high: float = math.inf
    ):
        self.shape = shape
        self.low = low
        self.high = high


class Simulation:
    """Pathmind's Python interface for multiple agents. Make sure to initialize
    all parameters you need for your simulation here, so that e.g. the `reset`
    method can restart a new simulation.

    The "action" value below is a per-agent dictionary. If your action_space returns
    a single value for agent 0, then action[0] will be a float value, otherwise
    a numpy array with specified shape. You use "action" to apply the next actions
    to your agents in the "step" function.
    """

    action: Dict[int, Union[float, np.ndarray]] = None

    def __init__(self, *args, **kwargs):
        """Set any properties and initial states needed for your simulation."""

    def set_action(self, action: Dict[int, Union[float, np.ndarray]]):
        """Use this to test your own decisions, or to integrate with Pathmind's Policy Server.
        set_action should always be executed before running the next step of your simulation."""
        self.action = action
        raise NotImplementedError

    def number_of_agents(self) -> int:
        """Returns the total number of agents to be controlled by Pathmind."""
        raise NotImplementedError

    def action_space(self, agent_id: int) -> Union[Continuous, Discrete]:
        """Return a Discrete or Continuous action space per agent."""
        raise NotImplementedError

    def step(self) -> None:
        """Carry out all things necessary at the next time-step of your simulation,
        in particular update the state of it. You have access to 'self.action', as
        explained above."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset your simulation parameters."""
        raise NotImplementedError

    def get_reward(self, agent_id: int) -> Dict[str, float]:
        """Get reward terms as a dictionary, given the current simulation state
        per agent, which are the foundation of creating reward functions in Pathmind."""
        raise NotImplementedError

    def get_observation(self, agent_id: int) -> Dict[str, Union[float, List[float]]]:
        """Get a dictionary of observations for the current state of the simulation per agent. Each
        observation can either be a single numeric value or a list thereof."""
        raise NotImplementedError

    def is_done(self, agent_id: int) -> bool:
        """Has this agent reached its target?"""
        raise NotImplementedError

    def run(
        self,
        policy,
        out_csv: Optional[str] = None,
        num_episodes: int = 1,
        sleep: Optional[int] = None,
    ) -> None:
        """
        Runs a simulation with a given policy. In Reinforcement Learning terms this creates a
        "rollout" of the policy over the specified number of episodes to run in the simulation.

        :param policy: A Pathmind Policy (local, server, or random)
        :param out_csv: If you specify an output CSV file, results will be stored there for debugging purposes.
        :param num_episodes: the number of episodes to run rollouts for.
        :param sleep: Optionally sleep for "sleep" seconds to make debugging easier.
        """
        # Only debug single episodes
        debug_mode = True if num_episodes == 1 else False
        done = False
        self.reset()
        agents = range(self.number_of_agents())
        table = PrettyTable()
        table.field_names = (
            ["Episode", "Step"]
            + [f"observations_{i}" for i in agents]
            + [f"actions_{i}" for i in agents]
            + [f"rewards_{i}" for i in agents]
            + [f"done_{i}" for i in agents]
        )
        print(table)

        for episode in range(num_episodes):
            step = 0
            while not done:
                row = [episode, step]
                if sleep:
                    # Optionally sleep for "sleep" seconds for easier debugging.
                    time.sleep(sleep)

                # Observations are "initial", i.e. before the action
                row += [self.get_observation(agent_id) for agent_id in agents]

                actions = policy.get_actions(self)
                self.action = actions

                self.step()
                dones = {
                    f"agent_{agent_id}": self.is_done(agent_id) for agent_id in agents
                }

                row += [self.action[agent_id] for agent_id in agents]
                row += [self.get_reward(agent_id) for agent_id in agents]
                row += [d for d in dones.values()]
                table.add_row(row)

                step += 1
                done = all(dones.values())

                if debug_mode:
                    print(row)

            if debug_mode:
                print(table)

            table_string = table.get_string()
            result = [
                tuple(filter(None, map(str.strip, splitline)))
                for line in table_string.splitlines()
                for splitline in [line.split("|")]
                if len(splitline) > 1
            ]

            if out_csv:
                with open(out_csv, "w") as out:
                    writer = csv.writer(out)
                    writer.writerows(result)

    def train(self, base_folder: str = "./", observation_yaml: str = None):
        """
        :param base_folder the path to your base folder containing all your Python code. Defaults to the current
            working directory, which assumes you start training from the base of your code base.
        :param observation_yaml: optional string with path to an observation yaml
        """

        env_name = str(self.__class__).split("'")[1]
        multi_agent = self.number_of_agents() > 1

        if not observation_yaml:
            write_observation_yaml(self, base_folder)
            obs_yaml = os.path.join(base_folder, "obs.yaml")
        else:
            obs_yaml = observation_yaml

        token = os.environ.get("PATHMIND_TOKEN")
        if not token:
            raise ValueError(
                "No Pathmind API token specified, "
                "please export 'PATHMIND_TOKEN' as environment variable."
            )

        shutil.make_archive("training", "zip", base_folder)

        cmd = f"""curl -i -XPOST \
              -H "X-PM-API-TOKEN: {token}" \
              -F 'file=@training.zip' \
              -F 'isPathmindSimulation=true' \
              -F 'env={env_name}' \
              -F 'start=true' \
              -F 'multiAgent={multi_agent}' \
              -F 'obsSelection={obs_yaml}' \
              https://api.pathmind.com/py/upload
            """

        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        if b"201" not in result.stdout:
            print(result.stdout)
            raise Exception("Could not start training, check the message above.")

        for line in result.stdout.split(b"\r\n"):
            if b"location" in line:
                location = line.split(b": ")[-1]
                print(f">>> See your Pathmind experiment at: \n\t{location.decode()}")

        return result


def write_observation_yaml(simulation: Simulation, folder) -> None:
    """Writes a YAML file with observation names that will be used by
    the training program to select which observation values are used for
    training.

    :param simulation: A Pathmind Simulation
    :param folder: the local folder which should contain the "obs.yaml" file
    """
    obs_name_list = list(simulation.get_observation(0).keys())
    obs = {"observations": obs_name_list}

    obs_file = os.path.join(folder, "obs.yaml")
    with open(obs_file, "w+") as f:
        f.write(yaml.dump(obs))
