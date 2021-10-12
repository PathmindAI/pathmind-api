from simulation import Simulation
from policy import server_action, local_action, random_action
import os
import yaml
import pprint
import numpy as np

def write_observation_yaml(simulation: Simulation, file_path) -> None:
    obs_name_list = list(simulation.get_observation(0).keys())
    obs = {"observations": obs_name_list}
    with open(os.path.join(file_path, "obs.yaml"), "w") as f:
        f.write(yaml.dump(obs))

def rollout_episodes(simulation, policy_file=None, server_url=None, debug_mode=True, episodes=1):
    pp = pprint.PrettyPrinter(indent=4)
    debug_mode = False if episodes > 1 else debug_mode
    done = False
    simulation.reset()
    agents = range(simulation.number_of_agents())

    for episode in range(episodes):
        print(f"Episode : {episode + 1}")
        step = 0
        while True:
            if debug_mode:
                print("----------------------------")
                print(f"Step : {step + 1}")
                print("----------------------------")
                pp.pprint({"Initial observations": {f"agent_{agent_id}": simulation.get_observation(agent_id) for agent_id in agents}})

            if policy_file:
                actions = local_action(simulation, policy_file)
            elif server_url:
                actions = server_action(simulation, server_url)
            else: #random action
                choices = simulation.action_space(0).choices
                size = simulation.action_space(0).size
                actions = np.random.randint(choices, size=(len(agents), size))

            simulation.action = actions

            simulation.step()
            dones = {f"agent_{agent_id}": simulation.is_done(agent_id) for agent_id in agents}
            if debug_mode:
                pp.pprint({"Actions": {f"agent_{agent_id}": simulation.action[agent_id] for agent_id in agents}})
                pp.pprint({"Rewards": {f"agent_{agent_id}": simulation.get_reward(agent_id) for agent_id in agents}})
                pp.pprint({"Next observations": {f"agent_{agent_id}": simulation.get_observation(agent_id) for agent_id in agents}})
                pp.pprint({"Done": dones})
            step += 1

            if all(dones.values()):
                break    

if __name__=="__main__":
    from pathmind.examples.mouse_env_pathmind import MouseAndCheese
    env = MouseAndCheese()
    rollout_episodes(env)
