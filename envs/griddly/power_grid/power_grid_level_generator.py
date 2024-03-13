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
        "altar:cooldown": [2, 5],
        "altar:cost": [10, 200],

        "charger:cooldown": [ 2, 2 ],
        "charger:energy": [ 100, 100 ],
        "generator:cooldown": [ 20, 100 ],

        "agent:energy:regen": [0, 0],
        "agent:energy:initial": [10, 200],
        "agent:energy:max": [500, 500],

        "gift:energy": [20, 20],

        "cost:move:predator": [0, 0],
        "cost:move:prey": [0, 0],
        "cost:move": [0, 0],
        "cost:jump": [3, 3],
        "cost:rotate": [0, 0],
        "cost:shield": [0, 2],
        "cost:shield:upkeep": [1, 2],
        "cost:frozen": [0, 0],

        "cost:attack": [5, 40],
        "attack:damage": [5, 40],
        "attack:freeze_duration": [5, 100],
    }

    LEVEL_CONFIG = {
        "tile_size": [32, 32],
        "width": [10, 20],
        "height": [10, 20],
        "wall_density": [0.1, 0.3],
        "milestone_density": [0, 0.1],
        "num_altars": [1, 20],
        "num_chargers": [5, 20],
        "num_generators": [5, 50],
        "wall_density": [0.0, 0.15],

        "rsm_num_families": [0, 0],
        "rsm_family_reward": [0, 0],

        "reward_rank_steps": [10, 500],
        "reward_prestige_weight": [10, 50],
    }

    def __init__(self, cfg):
        """
        Args:
            cfg: Optional configuration object.
        """
        self.cfg = cfg
        if isinstance(self.cfg, argparse.Namespace):
            self.cfg = vars(self.cfg)

        self.num_agents = self.cfg["env_num_agents"]
        self.max_steps = self.cfg["env_max_steps"]
        with open("./envs/griddly/power_grid/gdy/power_grid.yaml", encoding="utf-8") as file:
            self.game_config = yaml.safe_load(file)

        # make sure all the config variables are exist in the game config
        game_config_vars = set([
            v["Name"][5:] for v in self.game_config["Environment"]["Variables"]
            if v["Name"].startswith("conf:")])
        assert game_config_vars == set(self.GAME_CONFIG.keys()), \
            f" DIF : {game_config_vars - set(self.GAME_CONFIG.keys())}" \
            f" DIF : {set(self.GAME_CONFIG.keys()) - game_config_vars}"

    def make_env(self, render_mode="rgb_array"):
        def _update_global_variable(game_config, var_name, value):
            jmespath.search('Environment.Variables[?Name==`{}`][]'.format(var_name), game_config)[0]["InitialValue"] = value

        def _update_object_variable(game_config, object_name, var_name, value):
            jmespath.search('Objects[?Name==`{}`][].Variables[?Name==`{}`][]'.format(
                object_name, var_name), game_config)[0]["InitialValue"] = value

        game_config = deepcopy(self.game_config)
        game_config["Environment"]["Player"]["Count"] = self.num_agents
        game_config["Environment"]["Observers"]["GlobalSpriteObserver"]["TileSize"] = int(self.sample_cfg("tile_size"))
        game_config["Environment"]["Levels"] = [self.make_level_string()]
        for var_name, value in self.GAME_CONFIG.items():
            _update_global_variable(
                game_config,
                f"conf:{var_name}",
                int(self.sample_cfg(var_name)))

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
        width = int(self.sample_cfg("width"))
        height = int(self.sample_cfg("height"))

        level = np.array([["."]*width]*height).astype("U6") # 2-char unicode strings
        floor_tiles = [".", "o"]

        # make the bounding box
        level[0,:] = "W"
        level[-1,:] = "W"
        level[:,0] = "W"
        level[:,-1] = "W"

        # make the agents
        for i in range(self.num_agents):
            # level[4][2*i] = f"A{i+1}"
            while True:
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] in floor_tiles:
                    level[y][x] = f"A{i+1}"
                    break


        # make the altars
        for i in range(int(self.sample_cfg("num_altars"))):
            for _ in range(10):
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] in floor_tiles:
                    level[y][x] = "a"
                    break

        # make the chargers
        for i in range(int(self.sample_cfg("num_chargers"))):
            for _ in range(10):
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] in floor_tiles:
                    level[y][x] = "c"
                    break

        # make the generators
        for i in range(int(self.sample_cfg("num_generators"))):
            for _ in range(10):
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if level[y][x] in floor_tiles:
                    level[y][x] = "g"
                    break

        # make obstacles
        wall_density = self.sample_cfg("wall_density")

        for i in range(int(width*height*wall_density)):
            x = np.random.randint(1, width-1)
            y = np.random.randint(1, height-1)
            if level[y][x] in floor_tiles:
                level[y][x] = "W"
        return level

    def sample_cfg(self, key):
        vals = self.cfg.get(
            "env_" + key,
            self.GAME_CONFIG.get(key, self.LEVEL_CONFIG.get(key)))
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

