from argparse import FileType
from calendar import c
from collections.abc import Callable, Iterable, Sequence
from typing import Optional
from griddly.gym import GymWrapper
import numpy as np
import yaml
import argparse
import typing

from envs.predictive_reward_env_wrapper import PredictiveRewardEnvWrapper
import envs.args_parsing as args_parsing

GYM_ENV_NAME = "GDY-Forage"

def get_value_possibly_from_range(vals: typing.List, range_selection_fn: typing.Callable):
    """
        Args: 
            vals: a list of 1 or 2 values
            range_selection_fn: a function that selects a value based on a (low, high) range boundaries
    """
    if len(vals) == 1:
        return lambda: vals[0]
    elif len(vals) == 2:
        return lambda: range_selection_fn(vals[0], vals[1])
    else:
        raise ValueError(f"Length of values list should be at most 2. Got: {len(vals)}")

class ForageEnvFactory:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.game_config = yaml.safe_load(open("./envs/griddly/forage/forage.yaml"))
        self.num_agents = self.cfg.forage_num_agents
        
        self.level_width_sampler = get_value_possibly_from_range(self.cfg.forage_width, np.random.randint)
        self.level_height_sampler = get_value_possibly_from_range(self.cfg.forage_height, np.random.randint)
        self.level_wall_density_sampler = get_value_possibly_from_range(self.cfg.forage_wall_density, np.random.uniform)

        if self.game_config["Environment"]["Player"]["Count"] != self.num_agents:
            self.game_config["Environment"]["Player"]["Count"] = self.num_agents
            self.game_config["Environment"]["Levels"][0] = self.make_level_string()

    def make(self):
        return PredictiveRewardEnvWrapper(
            GymWrapper(
                yaml_string=yaml.dump(self.game_config),
                player_observer_type="VectorAgent",
                global_observer_type="GlobalSpriteObserver",
                level=0,
                max_steps=self.cfg.forage_max_env_steps,
            ),
            prediction_error_reward=self.cfg.forage_prediction_error_reward,
        )


    def make_level_string(self):
        return "\n".join(["  ".join(row) for row in self._make_level()])

    def _make_level(self):
        width = self.level_width_sampler()
        height = self.level_height_sampler()
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
        wall_density = self.level_wall_density_sampler()
        
        for i in range(int(width*height*wall_density)):
            x = np.random.randint(1, width-1)
            y = np.random.randint(1, height-1)
            if level[y][x] == ".":
                level[y][x] = "W"
        return level

def add_env_args(parser: argparse.ArgumentParser) -> None:
    p = parser

    p.add_argument("--forage_num_agents", default=8, type=int, help="number of agents in the environment")

    p.add_argument("--forage_width",
                   default=15,
                   help='Level width. Can be either a integer value OR a low:high range (e.g., "10:20" to sample from [10, 20) - high excluded)',
                   action=args_parsing.PossiblyNumericRange2Number,
                   str2numeric_cast_fn=int)

    p.add_argument("--forage_height",
                   default=15,
                   help='Level height. Can be either a single integer OR a low:high range (e.g., "10:20" to sample from [10, 20) - high excluded)',
                   action=args_parsing.PossiblyNumericRange2Number,
                   str2numeric_cast_fn=int)

    p.add_argument("--forage_energy_per_agent", default=10, type=int)

    p.add_argument("--forage_wall_density",
                   default=0.05,
                   help='Level wall density. Can be either a single float OR a low:high range (e.g., "0.2:0.5") from which a value is uniformly drawn',
                   action=args_parsing.PossiblyNumericRange2Number,
                   str2numeric_cast_fn=float)

    p.add_argument("--forage_max_env_steps", default=1000, type=int)
    p.add_argument("--forage_prediction_error_reward", default=0.00, type=float)


