from typing import Any, Dict

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
import yaml
from env.griddly.mettagrid.game_builder import MettaGridGameBuilder
from env.wrapper.feature_masker import FeatureMasker
from env.wrapper.kinship import Kinship
from env.wrapper.last_action_tracker import LastActionTracker
from env.wrapper.reward_tracker import RewardTracker
import pufferlib
from util.sample_config import sample_config
from env.mettagrid.mettagrid_c import MettaGrid
from puffergrid.wrappers.grid_env_wrapper import PufferGridEnv

class MettaGridEnv(pufferlib.PufferEnv):
    def __init__(self, render_mode: str, **cfg):
        super().__init__()

        self._render_mode = render_mode
        self._cfg = OmegaConf.create(cfg)

        self.make_env()

    def make_env(self):
        game_cfg = OmegaConf.create(sample_config(self._cfg.game))
        self._game_builder = MettaGridGameBuilder(**game_cfg)
        level = self._game_builder.level()
        self._c_env = MettaGrid(game_cfg, level)
        self._grid_env = self._c_env

        # self._grid_env = PufferGridEnv(self._c_env)
        env = self._grid_env

        self._env = env
        #self._env = LastActionTracker(self._grid_env)
        #self._env = Kinship(**sample_config(self._cfg.kinship), env=self._env)
        #self._env = RewardTracker(self._env)
        #self._env = FeatureMasker(self._env, self._cfg.hidden_features)

    def reset(self, **kwargs):
        self.make_env()
        if hasattr(self, "buf"):
            self._c_env.set_buffers(
                self.buf.observations,
                self.buf.terminals,
                self.buf.truncations,
                self.buf.rewards)

        # obs, infos = self._env.reset(**kwargs)
        # self._compute_max_energy()
        # return obs, infos
        obs, infos = self._c_env.reset()
        return obs, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self._c_env.step(actions.astype(np.int32))

        rewards = np.array(rewards) # xcxc / self._max_level_reward_per_agent
        if terminated.all() or truncated.all():
            stats = self._c_env.get_episode_stats()
            game_reward = stats["game_stats"].get("reward", 0)
            infos = {
                "reward.sum": game_reward,
                "reward.mean": game_reward / self._c_env.num_agents(),
            }

        return obs, list(rewards), terminated.all(), truncated.all(), infos

    def process_episode_stats(self, episode_stats: Dict[str, Any]):
        for agent_stats in episode_stats["agent_stats"]:
            extra_stats = {}
            for stat_name in agent_stats.keys():
                if stat_name.startswith("action_"):
                    extra_stats[stat_name + "_pct"] = agent_stats[stat_name] / self._grid_env.current_timestep


            #     for object in self._game_builder.object_configs.keys():
            #         if stat_name.startswith(f"stats_{object}_") and object != "agent":
            #             symbol = self._game_builder._objects[object].symbol
            #             num_obj = self._griddly_yaml["Environment"]["Levels"][0].count(symbol)
            #             if num_obj == 0:
            #                 num_obj = 1
            #             extra_stats[stat_name + "_pct"] = agent_stats[stat_name] / num_obj

            agent_stats.update(extra_stats)
            agent_stats.update(episode_stats["game_stats"])
            # agent_stats["level_max_energy"] = self._max_level_energy
            # agent_stats["level_max_energy_per_agent"] = self._max_level_energy_per_agent
            # agent_stats["level_max_reward_per_agent"] = self._max_level_reward_per_agent

    def _compute_max_energy(self):
        pass
        # num_generators = self._griddly_yaml["Environment"]["Levels"][0].count("g")
        # num_converters = self._griddly_yaml["Environment"]["Levels"][0].count("c")
        # max_resources = num_generators * min(
        #     self._game_builder.object_configs.generator.initial_resources,
        #     self._max_steps / self._game_builder.object_configs.generator.cooldown)

        # max_conversions = num_converters * (
        #     self._max_steps / self._game_builder.object_configs.converter.cooldown
        # )
        # max_conv_energy = min(max_resources, max_conversions) * \
        #     np.mean(list(self._game_builder.object_configs.converter.energy_output.values()))

        # initial_energy = self._game_builder.object_configs.agent.initial_energy * self._game_builder.num_agents

        # self._max_level_energy = max_conv_energy + initial_energy
        # self._max_level_energy_per_agent = self._max_level_energy / self._game_builder.num_agents

        # self._max_level_reward_per_agent = self._max_level_energy_per_agent


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
        return self._env.num_agents()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    @property
    def grid_features(self):
        return self._env.grid_features()

    @property
    def global_features(self):
        return []
