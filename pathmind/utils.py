from simulation import Simulation
from policy import Policy
import os
import yaml
import pprint


def write_observation_yaml(simulation: Simulation, file_path) -> None:
    obs_name_list = list(simulation.get_observation(0).keys())
    obs = {"observations": obs_name_list}
    with open(os.path.join(file_path, "obs.yaml"), "w") as f:
        f.write(yaml.dump(obs))


def rollout_episodes(simulation: Simulation, policy: Policy, episodes=1):
    pp = pprint.PrettyPrinter(indent=4)
    # Only debug single episodes
    debug_mode = True if episodes == 1 else False
    done = False
    simulation.reset()
    agents = range(simulation.number_of_agents())

    for episode in range(episodes):
        pp.pprint(f"Episode : {episode + 1}")
        step = 0
        while not done:
            if debug_mode:
                pp.pprint("----------------------------")
                pp.pprint(f"Step : {step + 1}")
                pp.pprint("----------------------------")
                pp.pprint(
                    {
                        "Initial observations": {
                            f"agent_{agent_id}": simulation.get_observation(agent_id)
                            for agent_id in agents
                        }
                    }
                )

            actions = policy.get_actions(simulation)
            simulation.action = actions

            simulation.step()
            dones = {
                f"agent_{agent_id}": simulation.is_done(agent_id) for agent_id in agents
            }
            if debug_mode:
                pp.pprint(
                    {
                        "Actions": {
                            f"agent_{agent_id}": simulation.action[agent_id]
                            for agent_id in agents
                        }
                    }
                )
                pp.pprint(
                    {
                        "Rewards": {
                            f"agent_{agent_id}": simulation.get_reward(agent_id)
                            for agent_id in agents
                        }
                    }
                )
                pp.pprint(
                    {
                        "Next observations": {
                            f"agent_{agent_id}": simulation.get_observation(agent_id)
                            for agent_id in agents
                        }
                    }
                )
                pp.pprint({"Done": dones})
            step += 1

            done = all(dones.values())
