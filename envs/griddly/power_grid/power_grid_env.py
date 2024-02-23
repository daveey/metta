import argparse
import enum
import random
import stat
from tkinter.font import families
from typing import List

import gymnasium as gym
import numpy as np
from pygame import init
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
        self._level_generator = level_generator
        self._render_mode = render_mode

        self._make_env()

        self.global_variable_names = sorted([
            v for v in self._griddly_env.game.get_global_variable_names()
              if v.startswith("conf:")])

        self._validate_griddly()
        self._max_level_energy = None

    def _validate_griddly(self):
        obj_order = self._griddly_env.game.get_object_names()
        var_order = self._griddly_env.game.get_object_variable_names()
        assert obj_order == sorted(obj_order)
        assert var_order == sorted(var_order)

    def _make_env(self):
        self._griddly_env = self._level_generator.make_env(self._render_mode)
        self._setup_reward_sharing()
        self._max_level_energy = self._compute_max_energy()

    def _setup_reward_sharing(self):
        self._reward_sharing_matrix = None
        num_families = int(self._level_generator.sample_cfg("rsm_num_families"))
        family_reward = self._level_generator.sample_cfg("rsm_family_reward")
        if num_families > 0 and family_reward > 0:
            agents = np.array(range(self._griddly_env.player_count))
            np.random.shuffle(agents)
            families = np.array_split(agents, num_families)

            rsm = np.zeros((len(agents), len(agents)), dtype=np.float32)
            # share the reward among the families
            for family in families:
                fr = family_reward / len(family)
                for a in family:
                    rsm[a, family] = fr
                    rsm[a, a] = 1 - fr

                # normalize
                rsm = rsm / rsm.sum(axis=1, keepdims=True)
                self._reward_sharing_matrix = rsm

    def render(self):
        return super().render()

    def reset(self, **kwargs):
        self._make_env()
        obs, infos = self._griddly_env.reset(**kwargs)
        self.global_variable_obs = np.array([
            v[0] for v in self._griddly_env.game.get_global_variable(self.global_variable_names).values()])
        return self._add_global_variables_obs(obs), infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self._griddly_env.step(actions)
        if terminated or truncated:
            self._add_episode_stats(infos)
        rewards = np.array(rewards) / 10.0
        if self._reward_sharing_matrix is not None:
            rewards = np.dot(self._reward_sharing_matrix, rewards)
        return self._add_global_variables_obs(obs), rewards, terminated, truncated, infos

    def _add_episode_stats(self, infos):
        stat_names = list(filter(
            lambda x: x.startswith("stats:"),
            self._griddly_env.game.get_global_variable_names()))
        stats = self._griddly_env.game.get_global_variable(stat_names)

        for agent in range(self._griddly_env.player_count):
            infos["episode_extra_stats"] = [{}] * self._griddly_env.player_count
            for stat_name in stat_names:
                # some are per-agent, some are just global {0: val}
                stat_val = stats[stat_name][0]
                if len(stats[stat_name]) > 1:
                    stat_val = stats[stat_name][agent + 1]
                infos["episode_extra_stats"][agent][stat_name] = stat_val
            infos["episode_extra_stats"][agent]["level_max_energy"] = self._max_level_energy
            infos["episode_extra_stats"][agent]["level_max_energy_per_agent"] = self._max_level_energy / self._griddly_env.player_count

    def _compute_max_energy(self):
        # compute the max possible energy for the level
        charger_energy, generator_cooldown, agent_regen = map(
            lambda x: float(x[0]),
            self._griddly_env.game.get_global_variable([
                "conf:charger:energy",
                "conf:generator:cooldown",
                "conf:agent:energy:regen"]
            ).values())

        num_steps = self._level_generator.max_steps
        num_generators = len(list(
            filter(lambda x: x["Name"] == "generator",
            self._griddly_env.game.get_state()["Objects"])))
        max_batteries = num_generators * (1 + num_steps // generator_cooldown)
        max_level_energy = max_batteries * charger_energy + self._griddly_env.player_count * agent_regen * num_steps

        return max_level_energy

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
            }) for o in self._griddly_env.observation_space]

    @property
    def action_space(self):
        return self._griddly_env.action_space

    @property
    def global_observation_space(self):
        return self._griddly_env.global_observation_space

    @property
    def player_count(self):
        return self._griddly_env.player_count


    def render_observer(self, *args, **kwargs):
        return self._griddly_env.render_observer(*args, **kwargs)
