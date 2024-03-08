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
        self._max_level_energy_per_agent = None
        self._episode_rewards = None
        self._prestige_reward_weight = None
        self._prestige_steps = None
        self._episode_prestige_rewards = None
        self._last_actions = None
        self._last_rewards = None
        self._reward_sharing_matrix = None
        self._agent_kinship = None

    def _validate_griddly(self):
        obj_order = self._griddly_env.game.get_object_names()
        var_order = self._griddly_env.game.get_object_variable_names()
        assert obj_order == sorted(obj_order)
        assert var_order == sorted(var_order)
        assert self._griddly_env.action_names == ["rotate", "move", "use", "shield", "attack"]

    def _make_env(self):
        self._griddly_env = self._level_generator.make_env(self._render_mode)
        self._max_steps = self._level_generator.max_steps
        self._setup_reward_sharing()
        self._compute_max_energy()
        self._episode_rewards = np.array([0] * self._griddly_env.player_count, dtype=np.float32)
        self._prestige_steps = int(self._level_generator.sample_cfg("reward_rank_steps"))
        self._prestige_reward_weight = self._level_generator.sample_cfg("reward_prestige_weight")

    def _setup_reward_sharing(self):
        self._reward_sharing_matrix = None
        num_families = int(self._level_generator.sample_cfg("rsm_num_families"))
        family_reward = self._level_generator.sample_cfg("rsm_family_reward")

        self._agent_kinship = np.array(range(self._griddly_env.player_count))

        if num_families > 0 and family_reward > 0:
            agents = np.array(range(self._griddly_env.player_count))
            np.random.shuffle(agents)
            families = np.array_split(agents, num_families)

            rsm = np.zeros((len(agents), len(agents)), dtype=np.float32)
            # share the reward among the families
            for family_id, family in enumerate(families):
                fr = family_reward / ( len(family) - 1 )
                for a in family:
                    self._agent_kinship[a] = family_id
                    rsm[a, family] = fr
                    rsm[a, a] = 1 - family_reward

            # normalize
            rsm = rsm / rsm.sum(axis=1, keepdims=True)
            self._reward_sharing_matrix = rsm

    def render(self):
        return super().render()


    def reset(self, **kwargs):
        self._make_env()
        obs, info = self._griddly_env.reset(**kwargs)
        if self._griddly_env.player_count == 1:
            obs = [obs]

        self._step = 0
        self._last_actions = np.zeros((self._griddly_env.player_count, 2), dtype=np.int32)
        self._last_rewards = np.zeros(self._griddly_env.player_count, dtype=np.float32)
        self._episode_prestige_rewards = np.array([0] * self._griddly_env.player_count, dtype=np.float32)
        self._global_variable_obs = np.array([
            v[0] for v in self._griddly_env.game.get_global_variable(self.global_variable_names).values()])

        augmented_obs = self._augment_observations(obs)
        if self._griddly_env.player_count == 1:
            return augmented_obs[0], info
        else:
            return self._augment_observations(obs), info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self._griddly_env.step(actions)
        if self._griddly_env.player_count == 1:
            self._last_actions = [actions]
            obs = [obs]
            rewards = [rewards]
        else:
            self._last_actions = actions

        self._step += 1
        rewards = np.array(rewards, dtype=np.float32)

        # prestige rewards
        rewards = self._compute_presitge_rewards(rewards)

        # share the rewards
        if self._reward_sharing_matrix is not None:
            rewards = self._reward_sharing_matrix @ rewards.transpose()

        # normalize by total available energy
        rewards /= self._max_level_energy_per_agent

        if terminated or truncated:
            self._add_episode_stats(info)

        self._last_rewards = rewards
        augmented_obs = self._augment_observations(obs)
        if self._griddly_env.player_count == 1:
            return augmented_obs[0], rewards[0], terminated, truncated, info
        else:
            return augmented_obs, rewards, terminated, truncated, info

    def _compute_presitge_rewards(self, rewards):

        if self._prestige_reward_weight == 0 or self._step % self._prestige_steps != 0:
            return rewards

        altar_energy = np.array(
            [v for v in self._griddly_env.game.get_global_variable(
                ["stats:energy:used:altar"])["stats:energy:used:altar"].values()
            ])
        if altar_energy.sum() == 0:
            return rewards

        # rank the agents by their rewards
        rank = np.argsort(np.argsort(altar_energy))
        # Scale the ranks to the range -1 to 1
        prestige_rewards = 2.0 * (rank / (rank.size - 1)) - 1.0
        # normalize based on how often we get prestige rewards
        prestige_rewards *= self._prestige_reward_weight * self._prestige_steps / self._max_steps
        self._episode_prestige_rewards += prestige_rewards
        rewards += prestige_rewards
        return rewards

    def _add_episode_stats(self, infos):
        stat_names = list(filter(
            lambda x: x.startswith("stats:"),
            self._griddly_env.game.get_global_variable_names()))
        stats = self._griddly_env.game.get_global_variable(stat_names).copy()
        infos["episode_extra_stats"] = []

        for agent in range(self._griddly_env.player_count):
            agent_stats = {}
            for stat_name in stat_names:
                # some are per-agent, some are just global {0: val}
                stat_val = stats[stat_name][0]
                if len(stats[stat_name]) > 1:
                    stat_val = stats[stat_name][agent + 1]
                    if stat_name.startswith("stats:action"):
                        stat_val /= self._max_steps
                agent_stats[stat_name] = stat_val
            agent_stats["prestige_reward"] = self._episode_prestige_rewards[agent]
            agent_stats["level_max_energy"] = self._max_level_energy
            agent_stats["level_max_energy_per_agent"] = self._max_level_energy_per_agent
            infos["episode_extra_stats"].append(agent_stats)

    def _compute_max_energy(self):
        # compute the max possible energy for the level
        charger_energy, generator_cooldown, agent_regen = map(
            lambda x: float(x[0]),
            self._griddly_env.game.get_global_variable([
                "conf:charger:energy",
                "conf:generator:cooldown",
                "conf:agent:energy:regen"]
            ).values())

        num_steps = self._max_steps
        num_generators = len(list(
            filter(lambda x: x["Name"] == "generator",
            self._griddly_env.game.get_state()["Objects"])))
        max_resources = num_generators * (1 + num_steps // generator_cooldown)
        max_charger_energy = float(charger_energy) * max_resources
        self._max_level_energy = max_charger_energy + self._griddly_env.player_count * agent_regen * num_steps
        self._max_level_energy_per_agent = self._max_level_energy / self._griddly_env.player_count

    def _augment_observations(self, obs):
        return [{
            "obs": agent_obs,
            "global_vars": self._global_variable_obs,
            "last_action": np.array(self._last_actions[agent]),
            "last_reward": np.array(self._last_rewards[agent]),
            "kinship": np.array(self._agent_kinship[agent]),
        } for agent, agent_obs in enumerate(obs)]


    @property
    def observation_space(self):
        augment = lambda o: gym.spaces.Dict({
            "obs": o,
            "global_vars": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=[len(self.global_variable_names)],
                dtype=np.float32),
            "last_action": gym.spaces.Box(
                low=0, high=255,
                shape=[2],
                dtype=np.int32),
            "last_reward": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=[1],
                dtype=np.float32),
            "kinship": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=[1],
                dtype=np.float32)
            })
        if self._griddly_env.player_count == 1:
            return augment(self._griddly_env.observation_space)
        else:
            return [augment(o) for o in self._griddly_env.observation_space]

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
