import argparse
from copy import deepcopy
from typing import List

import gymnasium as gym
import numpy as np
import yaml
from griddly.gym import GymWrapper
from gymnasium.core import Env

import util.args_parsing as args_parsing
import jmespath

class PowerGridLevelGenerator():
    GAME_CONFIG = {
        "battery:energy": [10, 100],
        "charger:cooldown": [10, 100],
        "energy:regen": [1, 2],
        "agent:initial_energy": [10, 200],

        "cost:move": [0, 2],
        "cost:rotate": [0, 2],
        "cost:pickup": [0, 2],
        "cost:use": [0, 2],
        "cost:shield": [0, 2],
        "cost:shield:upkeep": [0, 2],
        "cost:frozen": [0, 2],
        "cost:prestige": [30, 100],
        "cost:prestige:margin": [5, 20],
        "cost:attack": [5, 40],

        "attack:damage": [5, 40],
        "attack:freeze_duration": [5, 30],
    }

    LEVEL_CONFIG = {
        "width": [10, 20],
        "height": [10, 20],
        "wall_density": [0.1, 0.3],
        "chargers_per_agent": [1, 2],
        "wall_density": [0.0, 0.15],
    }

    def __init__(self, cfg):
        """
        Args:
            cfg: Optional configuration object.
        """
        self.cfg = cfg
        self.num_agents = self.cfg.env_num_agents
        self.max_steps = self.cfg.env_max_steps
        with open("./envs/griddly/power_grid/gdy/power_grid.yaml", encoding="utf-8") as file:
            self.game_config = yaml.safe_load(file)

        # make sure all the config variables are exist in the game config
        game_config_vars = set([
            v["Name"][5:] for v in self.game_config["Environment"]["Variables"]
            if v["Name"].startswith("conf:")])
        assert game_config_vars == set(self.GAME_CONFIG.keys()), \
            f"game_config_vars: {game_config_vars}, GAME_CONFIG: {self.GAME_CONFIG.keys()}"

    def make_env(self, render_mode="rgb_array"):
        def _update_global_variable(game_config, var_name, value):
            jmespath.search('Environment.Variables[?Name==`{}`][]'.format(var_name), game_config)[0]["InitialValue"] = value

        # def _update_object_variable(game_config, object_name, var_name, value):
        #     jmespath.search('Objects[?Name==`{}`][].Variables[?Name==`{}`][]'.format(
        #         object_name, var_name), game_config)[0]["InitialValue"] = value

        game_config = deepcopy(self.game_config)
        game_config["Environment"]["Player"]["Count"] = self.num_agents
        game_config["Environment"]["Levels"] = [self.make_level_string()]
        for var_name, value in self.GAME_CONFIG.items():
            _update_global_variable(
                game_config,
                f"conf:{var_name}",
                int(sample_value(self.cfg.__dict__.get(f"env_{var_name}", value))))

        env = GymWrapper(
            yaml_string=yaml.dump(game_config),
            player_observer_type="VectorAgent",
            global_observer_type="GlobalSpriteObserver",
            level=0,
            max_steps=self.max_steps,
            render_mode=render_mode,
        )
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
        width = int(sample_value(self.cfg.env_width))
        height = int(sample_value(self.cfg.env_height))
        num_chargers = int(self.num_agents * sample_value(self.cfg.env_chargers_per_agent))

        # make the bounding box
        level = [["."] * width for _ in range(height)]
        level[0] = ["W"] * width
        level[-1] = ["W"] * width
        for i in range(height):
            level[i][0] = "W"
            level[i][-1] = "W"

        # make the agents
        for i in range(self.num_agents):
            # level[4][2*i] = f"A{i+1}"
            while True:
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] == ".":
                    level[y][x] = f"A{i+1}"
                    break

        # make the energy
        for i in range(num_chargers):
            # level[40][2*i] = f"o"
            for _ in range(10):
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] == ".":
                    level[y][x] = "o"
                    break

        # make obstacles
        wall_density = sample_value(self.cfg.env_wall_density)

        for i in range(int(width*height*wall_density)):
            x = np.random.randint(1, width-1)
            y = np.random.randint(1, height-1)
            if level[y][x] == ".":
                level[y][x] = "W"
        return level

def sample_value(vals):
    if len(vals) == 1:
        return vals[0]
    elif len(vals) == 2:
        return np.random.uniform(vals[0], vals[1])
    raise ValueError(f"Length of values list should be at most 2. Got: {len(vals)}")

def add_env_args(parser: argparse.ArgumentParser) -> None:
    p = parser
    for k, v in PowerGridLevelGenerator.GAME_CONFIG.items():
        p.add_argument(f"--env_{k}",
            default=[v[0], v[1]],
            help=f'{k}. Can be either a single float OR a low:high range (e.g., "0.2:0.5") from which a value is uniformly drawn',
            action=args_parsing.PossiblyNumericRange2Number,
            str2numeric_cast_fn=float)

    for k, v in PowerGridLevelGenerator.LEVEL_CONFIG.items():
        p.add_argument(f"--env_{k}",
            default=[v[0], v[1]],
            help=f'{k}. Can be either a single float OR a low:high range (e.g., "0.2:0.5") from which a value is uniformly drawn',
            action=args_parsing.PossiblyNumericRange2Number,
            str2numeric_cast_fn=float)

