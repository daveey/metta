import argparse
import enum
import stat

import gymnasium as gym
import jmespath
import numpy as np
import yaml
from griddly.gym import GymWrapper
from gymnasium.core import Env
from envs.griddly.orb_world.orb_world_level_generator import OrbWorldLevelGenerator

import util.args_parsing as args_parsing
import cv2
import matplotlib.pyplot as plt
GYM_ENV_NAME = "GDY-OrbWorld"

class OrbWorldEnvWrapper(gym.Wrapper):
    def __init__(self, env: Env, level_generator: OrbWorldLevelGenerator):
        super().__init__(env)
        self.level_generator = level_generator

        obj_order = self.env.game.get_object_names()
        var_order = self.env.game.get_object_variable_names()
        # assert var_order == sorted(var_order)

        self.obs_order = list(range(len(obj_order) + len(var_order)))
        num_objects = len(obj_order)

        for expected, obj in enumerate(sorted(obj_order)):
            self.obs_order[expected] = obj_order.index(obj)
        for expected, var in enumerate(sorted(var_order)):
            self.obs_order[num_objects + expected] = num_objects + var_order.index(var)

    def reset(self, **kwargs):
        kwargs = kwargs or {}
        kwargs["options"] = {"level_string": self.level_generator.make_level_string()}
        obs, infos = self.env.reset(**kwargs)
        return self._reorder_obs(obs), infos

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
        return self._reorder_obs(obs), rewards, terminated, truncated, infos

    def _reorder_obs(self, obs):
        return [ob[self.obs_order] for ob in obs]

    @staticmethod
    def make_env(cfg, level_generator, render_mode="rgb_array"):
        """
        Creates a new instance of the OrbWorld environment.

        Returns:
            An instance of the OrbWorld environment.
        """
        with open("./envs/griddly/orb_world/gdy/orb_world.yaml", encoding="utf-8") as file:
            game_config = yaml.safe_load(file)

        game_config["Environment"]["Player"]["Count"] = cfg.env_num_agents
        game_config["Environment"]["Levels"] = [level_generator.make_level_string()]
        init_energy = level_generator.sample_initial_energy()
        _update_global_variable(game_config, "conf:agent:initial_energy", init_energy)

        env = OrbWorldEnvWrapper(GymWrapper(
                yaml_string=yaml.dump(game_config),
                player_observer_type="VectorAgent",
                global_observer_type="GlobalSpriteObserver",
                level=0,
                max_steps=cfg.env_max_steps,
                render_mode=render_mode,
            ), level_generator)
        return env

def _update_global_variable(game_config, var_name, value):
    jmespath.search('Environment.Variables[?Name==`{}`][]'.format(var_name), game_config)[0]["InitialValue"] = value

def _update_object_variable(game_config, object_name, var_name, value):
    jmespath.search('Objects[?Name==`{}`][].Variables[?Name==`{}`][]'.format(
        object_name, var_name), game_config)[0]["InitialValue"] = value
