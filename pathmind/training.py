from pathmind.simulation import Simulation

import os
import yaml
import shutil
import subprocess


class PathmindTraining:

    def __init__(self, simulation: Simulation, reward_function_name=None, observation_yaml=None):

        self.env_name = str(simulation.__class__).split("'")[1]
        self.simulation = simulation
        self.base_folder = self.env_name.split(".")[0]
        self.multi_agent = self.simulation.number_of_agents() > 1
        self.reward_fct_name = reward_function_name

        if not observation_yaml:
            write_observation_yaml(self.simulation, self.base_folder)
            self.obs_yaml = os.path.join(self.base_folder, "obs.yaml")
        else:
            self.obs_yaml = observation_yaml

        self.token = os.environ.get("PATHMIND_TOKEN")
        if not self.token:
            raise ValueError("No Pathmind API token specified, "
                             "please export 'PATHMIND_TOKEN' as environment variable.")

        shutil.make_archive(self.base_folder, 'zip', self.base_folder)

    # TODO training does _not_ start automatically currently
    def run(self):
        cmd = f"""curl -i -XPOST \
          -H "X-PM-API-TOKEN: {self.token}" \
          -F 'file=@{self.base_folder}.zip' \
          -F 'is_pathmind_simulation=true' \
          -F 'env={self.env_name}' \
          -F 'start=true' \
          -F 'multi_agent=true' \
          -F 'obs_selection={self.obs_yaml}' \
        """

        if self.reward_fct_name:
            cmd += f"""
            -F 'rew_fct_name={self.reward_fct_name}' \
            """

        cmd += "https://api.pathmind.com/py/upload"

        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        if b"201" not in result.stdout:
            print(result.stdout)
            raise Exception("Could not start training, check the message above.")


def write_observation_yaml(simulation: Simulation, file_path) -> None:
    obs_name_list = list(simulation.get_observation(0).keys())
    obs = {"observations": obs_name_list}
    with open(os.path.join(file_path, "obs.yaml"), "w") as f:
        f.write(yaml.dump(obs))
