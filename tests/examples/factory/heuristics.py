from .base import EnergyFactory


class SimpleHeuristic(EnergyFactory):
    def __init__(self, config="./pathmind/factory/config.yaml"):
        super().__init__(config=config)

        self.prices = self.config.get("price_intervals")
        self.power_levels = self.config.get("power_levels")

    def set_action(self):
        self.action = self.power_levels[-1]
        for i in range(len(self.prices) - 1):
            if self.prices[i] < self.price <= self.prices[i + 1]:
                self.action = self.power_levels[i]
