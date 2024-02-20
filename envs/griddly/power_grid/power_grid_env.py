import argparse
import enum
import stat
from typing import List

import gymnasium as gym
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
        self.env = self.level_generator.make_env(self.render_mode)
        obj_order = self.env.game.get_object_names()
        var_order = self.env.game.get_object_variable_names()
        assert obj_order == sorted(obj_order)
        assert var_order == sorted(var_order)
        self.global_variable_names = sorted([
            v for v in self.env.game.get_global_variable_names()
              if v.startswith("conf:")])

    def render(self):
        return super().render()

    def reset(self, **kwargs):
        self.env = self.level_generator.make_env(self.render_mode)
        obs, infos = self.env.reset(**kwargs)
        self.global_variable_obs = np.array([
            v[0] for v in self.env.game.get_global_variable(self.global_variable_names).values()])
        return self._add_global_variables_obs(obs), infos

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
        return self._add_global_variables_obs(obs), rewards, terminated, truncated, infos

    def _add_global_variables_obs(self, obs):
        return [{
            "obs": agent_obs,
            "global_vars": self.global_variable_obs
        } for agent_obs in obs]

    @property
    def observation_space(self):
        # augment the observation space with the global variables
        return [
            gym.spaces.Dict({
                "obs": o,
                "global_vars": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=[len(self.global_variable_names)],
                    dtype=np.float32)
            }) for o in self.env.observation_space]

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def global_observation_space(self):
        return self.env.global_observation_space

    @property
    def player_count(self):
        return self.env.player_count


