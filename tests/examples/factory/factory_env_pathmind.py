import datetime
import typing

import pandas as pd
from ruamel import yaml

from pathmind.simulation import Continuous, Discrete, Simulation


class EnergyFactory(Simulation):

    # dynamic variables
    total_cost = 0.0
    total_production = 0.0
    cost_of_action = 0.0
    cell_control_power = 0.0
    price = 0.0
    number_changes_per_day = 0

    # RL variables
    steps = 0
    number_of_actions = 24
    action_multiplier = 10
    number_of_observations = 5
    action = 1
    reward = [0 for _ in range(10)]  # "after" reward variables
    previous_reward = [0 for _ in range(10)]  # "before" reward variables

    # auxiliary variables
    previous_cell_control_power = None
    previous_day_of_year = None

    def number_of_agents(self) -> int:
        return 1

    def action_space(self, agent_id) -> typing.Union[Continuous, Discrete]:
        return Discrete(self.number_of_actions)

    def __init__(
        self,
        energy_data="./data/prices.xls",
        config="./default_config.yaml",
        only_positive_prices=True,
    ):

        with open(config, "r") as f:
            yaml_str = f.read()
        config = yaml.load(yaml_str, Loader=yaml.RoundTripLoader)

        # Factory parameters from config
        self.config = config
        self.only_positive_prices = only_positive_prices
        self.weekly_production_target = config.get("weekly_production_target")
        self.buying_window = config.get("buying_window")
        self.max_changes_per_day = config.get("max_changes_per_day")
        self.price_update_window = config.get("price_update_window")
        self.normalize_to_window = 60 / self.buying_window
        self.historic_voltage_estimation = config.get("historic_voltage_estimation")

        self.start_time = datetime.datetime(2020, 5, 11, 8, 50, 0)
        self.current_time = self.start_time

        data = pd.read_excel(energy_data)
        self.data = data.sort_values(by="Date", ascending=True)
        if self.only_positive_prices:
            self.data.Price[self.data.Price < 0] = 0
        last_timestamp = list(self.data.Date)[-1]
        self.end_time = last_timestamp.to_pydatetime()

        weeks_between = (self.end_time - self.start_time).days / 7.0
        self.adjusted_target = weeks_between * self.weekly_production_target

    def reset(self):
        self.current_time = self.start_time
        self.reward = [0 for _ in range(10)]
        self.previous_reward = [0 for _ in range(10)]
        self.total_cost = 0.0
        self.total_production = 0.0

    def step(self) -> None:
        self.steps += 1
        self.previous_reward = self.reward

        # update state
        self.current_time += datetime.timedelta(minutes=self.buying_window)
        self.price = self.price_at_date_time(self.current_time)

        self.set_action()
        self.cell_control_power = self.action[0]

        self.production_step()
        self.estimate_cost_of_action()
        self.update_change_counter()

        self.reward = self.compute_reward_variables()

    def get_observation(self, agent_id) -> typing.Dict[str, float]:
        return {
            "current_time_minute": self.current_time.minute,
            "current_time_hour": self.current_time.hour,
            "current_day": self.day_of_year(),
            "current_month": self.current_time.month,
            "price": self.price,
        }

    def get_reward(self, agent_id) -> typing.Dict[str, float]:
        after = self.reward
        before = self.previous_reward

        reward = 0
        reward += after[3] * 100 if self.is_done(agent_id) else 0
        reward -= (after[4] - before[4]) * 100
        reward -= 10 if after[6] > 2000 else 0
        reward -= -1 if after[7] > self.max_changes_per_day else 0
        reward = 1 if after[8] <= 20 and after[9] == 230 else 0

        return {"reward": reward}

    def is_done(self, agent_id) -> bool:
        return self.current_time >= self.end_time

    def get_metrics(self, agent_id) -> typing.List[float]:
        return [
            self.total_production,
            self.total_cost,
            self.cost_of_action,
            self.cell_control_power,
        ]

    def compute_reward_variables(self) -> typing.List[float]:
        fulfillment = self.total_production / self.adjusted_target
        target_cost = 2000000.0
        cost_fraction = self.total_cost / target_cost

        return [
            self.total_cost,
            self.total_cost / self.total_production
            if self.total_production > 0.0
            else 0.0,
            self.total_production,
            1 if self.total_production > self.adjusted_target else 0,
            (10 * (1 - fulfillment)) ** 2,  # parabolic reward term
            (10 * (1 - cost_fraction)) ** 3
            if cost_fraction > 0.7
            else 0,  # begin penalizing after 70% of cost target
            self.cost_of_action,  # future 30-min cost of last action taken
            self.number_changes_per_day,  # limit max power changes per day
            self.price,  # Encourage policy to maximize output when electricity cost is low
            self.cell_control_power,
        ]

    def set_action(self) -> None:
        """Heuristics etc. can override this method."""

    def estimate_cost_of_action(self):
        sum_cost_of_action = 0
        for minutes in range(
            self.price_update_window, self.buying_window + 1, self.price_update_window
        ):
            action_price = self.price_at_date_time(
                self.current_time + datetime.timedelta(minutes=minutes)
            )
            sum_cost_of_action += (
                action_price * self.total_milli_watts() / self.normalize_to_window
            )
        self.cost_of_action = sum_cost_of_action

    def update_change_counter(self):
        if self.previous_day_of_year == self.day_of_year():
            if self.previous_cell_control_power != self.cell_control_power:
                self.number_changes_per_day += 1
            self.previous_cell_control_power = self.cell_control_power
        else:
            self.number_changes_per_day = 0
        self.previous_day_of_year = self.day_of_year()

    def production_step(self):
        product_in_grams = (
            self.cell_control_power * (1.22 / self.normalize_to_window) * 124 * 0.915
        )
        self.total_cost += (
            self.total_milli_watts() * self.price / self.normalize_to_window
        )
        self.total_production += product_in_grams / 1000.0

    def day_of_year(self):
        return self.current_time.timetuple().tm_yday

    def price_at_date_time(self, time) -> float:
        filtered = self.data[self.data.Date == time]
        return float(filtered.Price) if len(filtered) > 0 else 0.0

    def total_milli_watts(self) -> float:
        return self.historic_voltage_estimation * self.cell_control_power / 1000.0
