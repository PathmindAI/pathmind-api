from .simulation import Simulation, LocalPolicy, RandomActions, Policy
import os
import yaml

def write_observation_yaml(simulation: Simulation, file_path) -> None:
    obs_name_list = list(simulation.get_observation(0).keys())
    obs = {"observations": obs_name_list}
    with open(os.path.join(file_path, "obs.yaml"), "w") as f:
        f.write(yaml.dump(obs))

def rollout_episodes(simulation, policy_file=None, server_url=None, debug_mode=True, episodes=1):
    verbose = False if episodes > 1 else verbose
    done = False
    self.reset()
    step = 0
    while True:
        if debug_mode:
            print(f"initial observation : {self.get_observation()}")

        if policy_file:
            action = local_action(simulation, policy_file)
        elif server_url:
            action = server_action(simulation, server_url)
        else: #random action
            action = np.random.randint(self.action_space.choices, size=self.action_space.size)

        self.set_action(action)
        self.step()
        if debug_mode:
            print(f"action : {self.action}")
            print(f"reward : {self.get_reward()}")
            print(f"next observation : {self.get_observation()}")
        step += 1
        if self.is_done(0):
            break    
