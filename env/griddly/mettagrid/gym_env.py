from typing import Any, Dict

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
import yaml
from env.griddly.griddly_gym_env import GriddlyGymEnv
from env.griddly.mettagrid.game_builder import MettaGridGameBuilder
from env.wrapper.feature_masker import FeatureMasker
from env.wrapper.kinship import Kinship
from env.wrapper.last_action_tracker import LastActionTracker
from env.wrapper.reward_tracker import RewardTracker
from util.sample_config import sample_config

class MettaGridGymEnv(gym.Env):
    def __init__(self, render_mode: str, **cfg):
        super().__init__()

        self._render_mode = render_mode
        self._cfg = OmegaConf.create(cfg)

        self.make_env()

    def make_env(self):
        self._game_builder = MettaGridGameBuilder(**sample_config(self._cfg.game))

        griddly_yaml = self._game_builder.build()
        self._griddly_yaml = yaml.safe_load(griddly_yaml)
        self._griddly_env = GriddlyGymEnv(
            griddly_yaml,
            self._cfg.max_action_value,
            self._game_builder.obs_width,
            self._game_builder.obs_height,
            self._max_steps,
            self._game_builder.num_agents,
            self._render_mode
        )

        self._env = LastActionTracker(self._griddly_env)
        self._env = Kinship(**sample_config(self._cfg.kinship), env=self._env)
        self._env = RewardTracker(self._env)
        self._env = FeatureMasker(self._env, self._cfg.hidden_features)

    def reset(self, **kwargs):
        self.make_env()
        obs, infos = self._env.reset(**kwargs)
        self._compute_max_energy()
        return obs, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        rewards = np.array(rewards) / self._max_level_reward_per_agent

        if terminated or truncated:
            self.process_episode_stats(info["episode_extra_stats"])

        return obs, list(rewards), terminated, truncated, info

    def process_episode_stats(self, episode_stats: Dict[str, Any]):
        for agent_stats in episode_stats:
            extra_stats = {}
            for stat_name in agent_stats.keys():
                if stat_name.startswith("stats_action_"):
                    extra_stats[stat_name + "_pct"] = agent_stats[stat_name] / self._griddly_env.num_steps
                for object in self._game_builder.object_configs.keys():
                    if stat_name.startswith(f"stats_{object}_") and object != "agent":
                        symbol = self._game_builder._objects[object].symbol
                        num_obj = self._griddly_yaml["Environment"]["Levels"][0].count(symbol)
                        if num_obj == 0:
                            num_obj = 1
                        extra_stats[stat_name + "_pct"] = agent_stats[stat_name] / num_obj

            agent_stats.update(extra_stats)
            agent_stats["level_max_energy"] = self._max_level_energy
            agent_stats["level_max_energy_per_agent"] = self._max_level_energy_per_agent
            agent_stats["level_max_reward_per_agent"] = self._max_level_reward_per_agent

    def _compute_max_energy(self):
        num_generators = self._griddly_yaml["Environment"]["Levels"][0].count("g")
        num_converters = self._griddly_yaml["Environment"]["Levels"][0].count("c")
        max_resources = num_generators * min(
            self._game_builder.object_configs.generator.initial_resources,
            self._max_steps / self._game_builder.object_configs.generator.cooldown)

        max_conversions = num_converters * (
            self._max_steps / self._game_builder.object_configs.converter.cooldown
        )
        max_conv_energy = min(max_resources, max_conversions) * \
            np.mean(list(self._game_builder.object_configs.converter.energy_output.values()))

        initial_energy = self._game_builder.object_configs.agent.initial_energy * self._game_builder.num_agents

        self._max_level_energy = max_conv_energy + initial_energy
        self._max_level_energy_per_agent = self._max_level_energy / self._game_builder.num_agents

        self._max_level_reward_per_agent = self._max_level_energy_per_agent


    @property
    def _max_steps(self):
        return self._game_builder.max_steps

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def player_count(self):
        return self._env.unwrapped.player_count

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    @property
    def grid_features(self):
        return self._griddly_env.grid_features

    @property
    def global_features(self):
        return self._griddly_env.global_features
