import argparse

import numpy as np
import yaml
from griddly.gym import GymWrapper
import gymnasium as gym
from gymnasium.core import Env, ObsType, WrapperObsType

GYM_ENV_NAME = "GDY-Mettaverse"

class MettaverseEnvWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        return self.env.step(actions)


class MettaverseEnvFactory:
    """
    Factory class for creating instances of the Forage environment.
    """

    def __init__(self, cfg=None):
        """
        Initializes a new instance of the ForageEnvFactory class.

        Args:
            cfg: Optional configuration object.
        """
        self.cfg = cfg
        with open("./envs/griddly/forage/forage.yaml", encoding="utf-8") as file:
            self.game_config = yaml.safe_load(file)
        self.num_agents = self.cfg.forage_num_agents

        if self.game_config["Environment"]["Player"]["Count"] != self.num_agents:
            self.game_config["Environment"]["Player"]["Count"] = self.num_agents
            self.game_config["Environment"]["Levels"][0] = self.make_level_string()

    def make(self):
        """
        Creates a new instance of the Forage environment.

        Returns:
            An instance of the Forage environment.
        """
        env = ForageEnvWrapper(GymWrapper(
                yaml_string=yaml.dump(self.game_config),
                player_observer_type="VectorAgent",
                global_observer_type="GlobalSpriteObserver",
                level=0,
                max_steps=self.cfg.forage_max_env_steps,
            ))
        return env

    def make_level_string(self):
        """
        Generates a string representation of the level configuration.

        Returns:
            A string representation of the level configuration.
        """
        return "\n".join(["  ".join(row) for row in self._make_level()])

    def _make_level(self):
        """
        Generates the level configuration.

        Returns:
            A 2D list representing the level configuration.
        """
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
                    level[y][x] = "o"
                    break

        # make obstacles
        for i in range(int(width*height*self.cfg.forage_wall_density)):
            x = np.random.randint(1, width-1)
            y = np.random.randint(1, height-1)
            if level[y][x] == ".":
                level[y][x] = "W"
        return level

def add_env_args(parser: argparse.ArgumentParser) -> None:
    """
    Add environment-specific arguments to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        None
    """
    p = parser

    p.add_argument("--forage_num_agents", default=8, type=int,
                   help="number of agents in the environment")

    p.add_argument("--forage_width_max", default=20, type=int, help="max level width")
    p.add_argument("--forage_width_min", default=10, type=int, help="min level width")

    p.add_argument("--forage_height_min", default=10, type=int, help="min level height")
    p.add_argument("--forage_height_max", default=20, type=int, help="max level height")

    p.add_argument("--forage_energy_per_agent", default=10, type=int)
    p.add_argument("--forage_wall_density", default=0.1, type=float)

    p.add_argument("--forage_max_env_steps", default=1000, type=int)
