
from math import inf
from typing import Any, Dict, List
from griddly.gym import GymWrapper
from griddly.wrappers.render_wrapper import RenderWrapper
import gymnasium as gym
import numpy as np
from functools import lru_cache

from omegaconf import OmegaConf
import yaml


class GriddlyEnv(gym.Env):
    def __init__(
            self,
            cfg: OmegaConf,
            griddly_yaml,
            num_agents: int,
            render_mode: str):

        self._cfg = cfg
        self._griddly_yaml = griddly_yaml
        self._num_agents = num_agents
        self._render_mode = render_mode
        self._num_actions = len(self._cfg.actions)
        self.num_steps = None

        self._global_env = RenderWrapper(
            self, "global",
            render_mode=render_mode
        )

    def make_env(self):
        env = GymWrapper(
            yaml_string=yaml.dump(self._griddly_yaml),
            player_observer_type="VectorAgent",
            global_observer_type="GlobalSpriteObserver",
            max_steps=self._cfg.max_steps,
            level=0,
            render_mode=self._render_mode,
        )
        self._validate(env)
        return env

    def reset(self, **kwargs):
        self._griddly_env = self.make_env()

        obs, info = self._griddly_env.reset(**kwargs)
        if self._num_agents == 1:
            obs = [obs]

        self.num_steps = 0
        self._compute_global_variable_obs()

        augmented_obs = self._augment_observations(obs)
        if self._num_agents == 1:
            return augmented_obs[0], info
        else:
            return self._augment_observations(obs), info

    def step(self, actions):
        actions = [
            a if a[0] < self._num_actions else [0, 0]
            for a in actions
        ]
        actions = [ a[:self._num_actions] for a in actions ]
        obs, rewards, terminated, truncated, info = self._griddly_env.step(actions)
        if self._num_agents == 1:
            obs = [obs]

        # config variables get update in the first few steps (species get set)
        if self.num_steps < 2:
            self._compute_global_variable_obs()

        self.num_steps += 1
        rewards = np.array(rewards, dtype=np.float32)

        if terminated or truncated:
            self._add_episode_stats(info)

        augmented_obs = self._augment_observations(obs)
        if self._num_agents == 1:
            return augmented_obs[0], rewards[0], terminated, truncated, info
        else:
            return augmented_obs, rewards, terminated, truncated, info

    def get_global_variables(self, var_names):
        return self._griddly_env.game.get_global_variable(var_names)

    def render(self, *args, **kwargs):
        return self._global_env.render()

    def process_episode_stats(self, episode_stats: Dict[str, Any]):
        return episode_stats

    def _compute_global_variable_obs(self):
        vals = []
        for v in self._griddly_env.game.get_global_variable(self._cfg.global_features).values():
            if len(v) == 1:
                vals.append([v[0]] * self._num_agents)
            else:
                vals.append(list(v.values())[1:])
        self._global_variable_obs = np.array(vals).transpose()

    def _validate(self, env):
        assert env.game.get_object_names() + \
              env.game.get_object_variable_names() == \
              self._cfg.grid_features, \
            f"Missing grid features: {set(self._cfg.grid_features) - set(env.game.get_object_names() + env.game.get_object_variable_names())}"

        assert env.action_names == self._cfg.actions, \
            f"Missing actions: {set(self._cfg.actions) - set(env.action_names)}"

        assert set(env.game.get_global_variable_names()).issuperset(
            self._cfg.global_features), \
            f"Missing global variables: {set(self._cfg.global_features) - set(env.game.get_global_variable_names())}"


        env_obs_space = env.observation_space
        env_act_space = env.action_space
        if self._num_agents > 1:
            env_obs_space = env_obs_space[0]
            env_act_space = env_act_space[0]

        assert self.observation_space["grid_obs"] == env_obs_space, \
            f"Grid observation space mismatch: {self.observation_space['grid_obs']} != {env_obs_space}"

        assert self.action_space == env_act_space, \
            f"Action space mismatch: {self.action_space} != {env_act_space}"

    def _add_episode_stats(self, infos):
        stat_names = list(filter(
            lambda x: x.startswith("stats:"),
            self._griddly_env.game.get_global_variable_names()))
        stats = self._griddly_env.game.get_global_variable(stat_names).copy()
        episode_stats = []

        for agent in range(self._num_agents):
            agent_stats = {}

            for stat_name in stat_names:
                # some are per-agent, some are just global {0: val}
                stat_val = stats[stat_name][0]
                if len(stats[stat_name]) > 1:
                    stat_val = stats[stat_name][agent + 1]
                agent_stats[stat_name] = stat_val

            agent_stats = {
                key.replace(':', '_'): value for key, value in agent_stats.items()
            }
            episode_stats.append(agent_stats)

        infos["episode_extra_stats"] = episode_stats

    def _augment_observations(self, obs):
        return [{
            "grid_obs": agent_obs,
            "global_vars": self._global_variable_obs[agent],
        } for agent, agent_obs in enumerate(obs)]

    @property
    def observation_space(self):
        agent_obs_space = gym.spaces.Dict({
            "global_vars": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=[ len(self._cfg.global_features) ],
                dtype=np.int32),
            "grid_obs": gym.spaces.Box(
                low=0, high=255,
                shape=[ len(self._cfg.grid_features), self._cfg.obs_width, self._cfg.obs_height],
                dtype=np.uint8
            )
        })

        return agent_obs_space

    @property
    def action_space(self):
        return gym.spaces.MultiDiscrete([len(self._cfg.actions), self._cfg.max_action_value])

    @lru_cache(maxsize=None)
    def global_observation_space(self, agent_id):
        return self._griddly_env.global_observation_space[agent_id]

    @property
    def player_count(self):
        return self._num_agents

    def render_observer(self, *args, **kwargs):
        return self._griddly_env.render_observer(*args, **kwargs)

