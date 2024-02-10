import argparse

import gymnasium as gym
import numpy as np
import yaml
from griddly.gym import GymWrapper
from gymnasium.core import Env

import util.args_parsing as args_parsing

class OrbWorldLevelGenerator():

    def __init__(self, cfg=None):
        """
        Args:
            cfg: Optional configuration object.
        """
        self.cfg = cfg
        self.num_agents = self.cfg.env_num_agents

        self.sample_level_width = args_parsing.get_value_possibly_from_range(self.cfg.orb_world_width, np.random.randint)
        self.sample_level_height = args_parsing.get_value_possibly_from_range(self.cfg.orb_world_height, np.random.randint)
        self.sample_level_wall_density = args_parsing.get_value_possibly_from_range(self.cfg.orb_world_wall_density, np.random.uniform)
        self.sample_num_factories_per_agent = args_parsing.get_value_possibly_from_range(self.cfg.orb_world_factories_per_agent, np.random.uniform)
        self.sample_initial_energy = args_parsing.get_value_possibly_from_range(self.cfg.orb_world_initial_energy, np.random.uniform)

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
        width = self.sample_level_width()
        height = self.sample_level_height()
        factories = int(self.num_agents * self.sample_num_factories_per_agent())

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
        for i in range(factories):
            for _ in range(10):
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] == ".":
                    level[y][x] = "o"
                    break

        # make obstacles
        wall_density = self.sample_level_wall_density()

        for i in range(int(width*height*wall_density)):
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


    p.add_argument("--orb_world_width",
                   default=[15],
                   help='Level width. Can be either a integer value OR a low:high range (e.g., "10:20" to sample from [10, 20) - high excluded)',
                   action=args_parsing.PossiblyNumericRange2Number,
                   str2numeric_cast_fn=int)

    p.add_argument("--orb_world_height",
                   default=[15],
                   help='Level height. Can be either a single integer OR a low:high range (e.g., "10:20" to sample from [10, 20) - high excluded)',
                   action=args_parsing.PossiblyNumericRange2Number,
                   str2numeric_cast_fn=int)

    p.add_argument("--orb_world_factories_per_agent",
                   default=[1],
                   help='Factories per agent.',
                   action=args_parsing.PossiblyNumericRange2Number,
                   str2numeric_cast_fn=float)

    p.add_argument("--orb_world_initial_energy",
                   default=[30],
                   help='Factories per agent.',
                   action=args_parsing.PossiblyNumericRange2Number,
                   str2numeric_cast_fn=int)

    p.add_argument("--orb_world_wall_density",
                   default=[0.05],
                   help='Level wall density. Can be either a single float OR a low:high range (e.g., "0.2:0.5") from which a value is uniformly drawn',
                   action=args_parsing.PossiblyNumericRange2Number,
                   str2numeric_cast_fn=float)

