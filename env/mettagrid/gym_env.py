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
from util.sample_config import sample_config
from env.mettagrid.mettagrid_c import MettaGrid
from env.puffergrid.grid_env import PufferGridEnv

class MettaGridGymEnv(gym.Env):
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
        self._grid_env = PufferGridEnv(
            self._c_env,
            num_agents=self._game_builder.num_agents,
            max_timesteps=self._max_steps,
            obs_width=self._game_builder.obs_width,
            obs_height=self._game_builder.obs_height)
        env = self._grid_env

        self._agent_layer = self._c_env.layer(env.type_ids.Agent)
        self._object_layer = self._c_env.layer(env.type_ids.Wall)

        for c in range(0, env._map_width):
            if env.grid_location_empty(0, c):
                env.add_object(env.type_ids.Wall, 0, c, self._object_layer)
            if env.grid_location_empty(env._map_height-1, c):
                env.add_object(env.type_ids.Wall, env._map_height-1, c, self._object_layer)

        for r in range(0, env._map_height):
            if env.grid_location_empty(r, 0):
                env.add_object(env.type_ids.Wall, r, 0, self._object_layer)
            if env.grid_location_empty(r, env._map_width-1):
                env.add_object(env.type_ids.Wall, r, env._map_width-1, self._object_layer)

        for agent_id in range(env._num_agents):
            while True:
                c = np.random.randint(0, env._map_width)
                r = np.random.randint(0, env._map_height)
                if env.grid_location_empty(r, c):
                    agent = env.add_agent(
                        env.type_ids.Agent, r, c, self._agent_layer,
                        id=agent_id
                    )
                    break

        for tree in range(50):
            c = np.random.randint(0, env._map_width)
            r = np.random.randint(0, env._map_height)
            if env.grid_location_empty(r, c):
                tree = env.add_object(env.type_ids.Tree, r, c, self._object_layer, has_food=1)

        self._env = self._grid_env
        #self._env = LastActionTracker(self._grid_env)
        #self._env = Kinship(**sample_config(self._cfg.kinship), env=self._env)
        #self._env = RewardTracker(self._env)
        #self._env = FeatureMasker(self._env, self._cfg.hidden_features)

    def reset(self, **kwargs):
        self.make_env()
        obs, infos = self._env.reset(**kwargs)
        self._compute_max_energy()
        return obs, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        rewards = np.array(rewards) # xcxc / self._max_level_reward_per_agent

        if terminated.all() or truncated.all():
            self.process_episode_stats(info["episode_stats"])
            info = {
                "episode_extra_stats": info["episode_stats"]["agent_stats"]
            }

        return obs, list(rewards), terminated.all(), truncated.all(), info

    def process_episode_stats(self, episode_stats: Dict[str, Any]):
        return # xcxc
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
        return self._env.unwrapped.player_count

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    @property
    def grid_features(self):
        return self._env.unwrapped.grid_features

    @property
    def global_features(self):
        return self._env.unwrapped.global_features
