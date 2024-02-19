import argparse
import enum
import stat
from typing import List

import gymnasium as gym
import jmespath
import numpy as np
import yaml
from griddly.gym import GymWrapper
from gymnasium.core import Env
from envs.griddly.power_grid.power_grid_level_generator import PowerGridLevelGenerator

import util.args_parsing as args_parsing
import cv2
import matplotlib.pyplot as plt
GYM_ENV_NAME = "GDY-PowerGrid"

class PowerGridEnv(gym.Env):
    def __init__(self, level_generator: PowerGridLevelGenerator, render_mode="rgb_array"):
        super().__init__()
        self.level_generator = level_generator
        self.render_mode = render_mode
        self.env = self._make_env()
        obj_order = self.env.game.get_object_names()
        var_order = self.env.game.get_object_variable_names()
        assert obj_order == sorted(obj_order)
        assert var_order == sorted(var_order)

    def render(self):
        return super().render()

    def reset(self, **kwargs):
        self.env = self._make_env()
        obs, infos = self.env.reset(**kwargs)
        return obs, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if terminated or truncated:
            stat_names = list(filter(
                lambda x: x.startswith("stats:"),
                self.env.game.get_global_variable_names()))
            stats = self.env.game.get_global_variable(stat_names)
            for agent in range(self.env.player_count):
                infos["episode_extra_stats"] = [{}] * self.env.player_count
                for stat_name in stat_names:
                    # some are per-agent, some are just global {0: val}
                    stat_val = stats[stat_name][0]
                    if len(stats[stat_name]) > 1:
                        stat_val = stats[stat_name][agent + 1]
                    infos["episode_extra_stats"][agent][stat_name] = stat_val
        rewards = np.array(rewards) / 10.0
        return obs, rewards, terminated, truncated, infos

    def _make_env(self):
        with open("./envs/griddly/power_grid/gdy/power_grid.yaml", encoding="utf-8") as file:
            game_config = yaml.safe_load(file)

        game_config["Environment"]["Player"]["Count"] = self.level_generator.num_agents
        game_config["Environment"]["Levels"] = [self.level_generator.make_level_string()]
        init_energy = self.level_generator.sample_initial_energy()
        _update_global_variable(game_config, "conf:agent:initial_energy", init_energy)

        env = GymWrapper(
            yaml_string=yaml.dump(game_config),
            player_observer_type="VectorAgent",
            global_observer_type="GlobalSpriteObserver",
            level=0,
            max_steps=self.level_generator.max_steps,
            render_mode=self.render_mode,
        )
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.global_observation_space = env.global_observation_space
        self.player_count = env.player_count

        return env

def _update_global_variable(game_config, var_name, value):
    jmespath.search('Environment.Variables[?Name==`{}`][]'.format(var_name), game_config)[0]["InitialValue"] = value

def _update_object_variable(game_config, object_name, var_name, value):
    jmespath.search('Objects[?Name==`{}`][].Variables[?Name==`{}`][]'.format(
        object_name, var_name), game_config)[0]["InitialValue"] = value
