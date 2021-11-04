import csv
import math
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Union

import numpy as np
import prettytable
import yaml
from gym import Env
from gym.spaces import Box as GymContinuous
from gym.spaces import Discrete as GymDiscrete
from or_gym import Env as OrEnv
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
        summary_csv: Optional[str] = None,
        num_episodes: int = 1,
        sleep: Optional[int] = None,
    ) -> None:
        """
        Runs a simulation with a given policy. In Reinforcement Learning terms this creates a
        "rollout" of the policy over the specified number of episodes to run in the simulation.

        :param policy: A Pathmind Policy (local, server, or random)
        :param out_csv: If you specify an output CSV file, complete results of the first episode will be stored there.
        :param summary_csv: If you specify a summary CSV file, a summary of reward terms over all episodes will be
            stored in that file.
        :param num_episodes: the number of episodes to run rollouts for.
        :param sleep: Optionally sleep for "sleep" seconds to make debugging easier.
        """
        # Only debug single episodes
        debug_mode = True if num_episodes == 1 else False
        done = False
        self.reset()

        agents = range(self.number_of_agents())
        table, summary = _define_tables(self, agents)

        for episode in range(num_episodes):

            step = 0
            done = False
            self.reset()
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

                dones = [self.is_done(agent_id) for agent_id in agents]
                row += [self.action[agent_id] for agent_id in agents]
                row += [self.get_reward(agent_id) for agent_id in agents]
                row += dones
                table.add_row(row)

                step += 1
                done = all(dones)

            # add reward terms in order after episode completion
            terms = [
                v for agent_id in agents for v in self.get_reward(agent_id).values()
            ]
            summary.add_row([episode] + terms)

            if debug_mode:
                print(">>> Complete table:\n")
                print(table)
                print(">>> Summary table:\n")
                print(summary)

            write_table(table=table, out_csv=out_csv)
            write_table(table=summary, out_csv=summary_csv)
            print(f"--------Finished episode {episode}--------")

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


def write_table(
    table: prettytable.prettytable.PrettyTable, out_csv: Optional[str]
) -> None:
    """Store a table to file

    :param table: a PrettyTable
    :param out_csv: the CSV file you want to store your results at.
    """
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


def _define_tables(simulation, agents):
    table = PrettyTable()
    table.field_names = (
        ["Episode", "Step"]
        + [f"observations_{i}" for i in agents]
        + [f"actions_{i}" for i in agents]
        + [f"rewards_{i}" for i in agents]
        + [f"done_{i}" for i in agents]
    )

    summary = PrettyTable()
    summary.field_names = ["Episode"] + [
        f"reward_{i}_{name}" for i in agents for name in simulation.get_reward(i).keys()
    ]

    return table, summary


def from_gym(gym_instance: Union[Env, OrEnv]) -> Simulation:
    """

    :param gym_instance: gym or OR-gym environment
    :return: A pathmind environment
    """

    class GymSimulation(Simulation):
        def __init__(self, gym_instance: Union[Env, OrEnv], *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.env = gym_instance
            self.observations: Dict[str, float] = {}
            self.rewards: Dict[str, float] = {"reward": 0}
            self.done: bool = False

        def number_of_agents(self) -> int:
            return 1

        def action_space(self, agent_id: int) -> Union[Continuous, Discrete]:
            gym_space = self.env.action_space
            if isinstance(gym_space, GymDiscrete):
                # TODO take care of MultiDiscrete.
                space = Discrete(choices=gym_space.n)
            elif isinstance(gym_space, GymContinuous):
                space = Continuous(
                    shape=gym_space.shape, low=gym_space.low, high=gym_space.high
                )
            else:
                raise ValueError(
                    f"Unsupported gym.spaces type {type(gym_space)}. Pathmind currently only allows"
                    f"gym.spaces.Discrete and gym.spaces.Box as valid action spaces."
                )
            return space

        def step(self) -> None:
            # This assumes "choices=1"
            action = self.action[0][0]
            obs, rew, done, _ = self.env.step(action)
            self.observations = {f"obs_{i}": o for i, o in enumerate(obs)}
            self.rewards = {"reward": rew}
            self.done = done

        def reset(self) -> None:
            obs = self.env.reset()
            self.observations = {f"obs_{i}": o for i, o in enumerate(obs)}
            self.done = False

        def get_reward(self, agent_id: int) -> Dict[str, float]:
            return self.rewards

        def get_observation(
            self, agent_id: int
        ) -> Dict[str, Union[float, List[float]]]:
            return self.observations

        def is_done(self, agent_id: int) -> bool:
            return self.done

    sim = GymSimulation(gym_instance=gym_instance)
    return sim
