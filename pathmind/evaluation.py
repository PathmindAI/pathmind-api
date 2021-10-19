from pathmind.simulation import Simulation
from pathmind.policy import Policy


from typing import Optional
import time
from prettytable import PrettyTable
import csv


def rollout_episodes(simulation: Simulation, policy: Policy, out_csv: Optional[str] = None,
                     episodes: int = 1, sleep: Optional[int] = None) -> None:
    # Only debug single episodes
    debug_mode = True if episodes == 1 else False
    done = False
    simulation.reset()
    agents = range(simulation.number_of_agents())
    table = PrettyTable()
    table.field_names = ["Episode", "Step"] + [f"observations_{i}" for i in agents] +\
                        [f"actions_{i}" for i in agents] + [f"rewards_{i}" for i in agents] +\
                        [f"done_{i}" for i in agents]

    for episode in range(episodes):
        step = 0
        while not done:
            row = [episode, step]
            if sleep:
                # Optionally sleep for "sleep" seconds for easier debugging.
                time.sleep(sleep)

            # Observations are "initial", i.e. before the action
            row += [simulation.get_observation(agent_id) for agent_id in agents]

            actions = policy.get_actions(simulation)
            simulation.action = actions

            simulation.step()
            dones = {f"agent_{agent_id}": simulation.is_done(agent_id) for agent_id in agents}

            row += [simulation.action[agent_id] for agent_id in agents]
            row += [simulation.get_reward(agent_id) for agent_id in agents]
            row += [d for d in dones.values()]
            table.add_row(row)

            step += 1
            done = all(dones.values())

        if debug_mode:
            print(table)

        table_string = table.get_string()
        result = [tuple(filter(None, map(str.strip, splitline))) for line in table_string.splitlines()
                  for splitline in [line.split("|")] if len(splitline) > 1]

        if out_csv:
            with open(out_csv, 'w') as out:
                writer = csv.writer(out)
                writer.writerows(result)
