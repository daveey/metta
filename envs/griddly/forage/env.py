from calendar import c
from typing import Optional
from griddly.gym import GymWrapper
import numpy as np
import yaml
import argparse

from envs.predictive_reward_env_wrapper import PredictiveRewardEnvWrapper

GYM_ENV_NAME = "GDY-Forage"

class ForageEnvFactory:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.game_config = yaml.safe_load(open("./envs/griddly/forage/forage.yaml"))
        self.num_agents = self.cfg.forage_num_agents

        if self.game_config["Environment"]["Player"]["Count"] != self.num_agents:
            self.game_config["Environment"]["Player"]["Count"] = self.num_agents
            self.game_config["Environment"]["Levels"][0] = self.make_level_string()

    def make(self):
        env = GymWrapper(
                yaml_string=yaml.dump(self.game_config),
                player_observer_type="VectorAgent",
                global_observer_type="GlobalSpriteObserver",
                level=0,
                max_steps=self.cfg.forage_max_env_steps,
            )

        return PredictiveRewardEnvWrapper(
            env,
            prediction_error_reward=self.cfg.forage_prediction_error_reward)

    def make_level_string(self):
        return "\n".join(["  ".join(row) for row in self._make_level()])

    def _make_level(self):
        width = np.random.randint(self.cfg.forage_width_min, self.cfg.forage_width_max)
        height = np.random.randint(self.cfg.forage_height_min, self.cfg.forage_height_max)
        energy = self.num_agents * self.cfg.forage_energy_per_agent

        # make the bounding box
        level = [["."] * width for _ in range(height)]
        level[0] = ["W"] * width
        level[-1] = ["W"] * width
        for i in range(height):
            level[i][0] = "W"
            level[i][-1] = "W"

        # make the agents
        for i in range(self.num_agents):
            while True:
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] == ".":
                    level[y][x] = f"A{i+1}"
                    break

        # make the energy
        for i in range(energy):
            for _ in range(10):
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] == ".":
                    level[y][x] = "e"
                    break

        # make obstacles
        for i in range(int(width*height*self.cfg.forage_wall_density)):
            x = np.random.randint(1, width-1)
            y = np.random.randint(1, height-1)
            if level[y][x] == ".":
                level[y][x] = "W"
        return level

def add_env_args(parser: argparse.ArgumentParser) -> None:
    p = parser

    p.add_argument("--forage_num_agents", default=8, type=int, help="number of agents in the environment")

    p.add_argument("--forage_width_max", default=20, type=int, help="max level width")
    p.add_argument("--forage_width_min", default=10, type=int, help="min level width")

    p.add_argument("--forage_height_min", default=10, type=int, help="min level height")
    p.add_argument("--forage_height_max", default=20, type=int, help="max level height")

    p.add_argument("--forage_energy_per_agent", default=10, type=int)
    p.add_argument("--forage_wall_density", default=0.1, type=float)

    p.add_argument("--forage_max_env_steps", default=1000, type=int)
    p.add_argument("--forage_prediction_error_reward", default=0.00, type=float)


