from pdb import set_trace as T
from types import SimpleNamespace
from typing import List
import numpy as np

import pettingzoo
import gymnasium as gym

from env.mettagrid import render
import pufferlib
from pufferlib.environment import PufferEnv

class PufferGridEnv(PufferEnv):
    def __init__(
            self,
            c_env_class,
            map_width=50,
            map_height=50,
            num_agents=1,
            max_timesteps=1000,
            obs_width=11,
            obs_height=11) -> None:

        super().__init__()
        self._map_width = map_width
        self._map_height = map_height
        self._num_agents = num_agents
        self._obs_width = obs_width
        self._obs_height = obs_height
        self._max_timesteps = max_timesteps

        self._c_env = c_env_class(
            map_width,
            map_height,
            num_agents,
            max_timesteps,
            obs_width,
            obs_height)

        self._buffers = SimpleNamespace(**{
            k: np.asarray(v) for k,v in self._c_env.get_buffers().items()
        })
        self._grid = np.asarray(self._c_env.get_grid())

        self.episode_rewards = np.zeros(num_agents, dtype=np.float32)
        self.dones = np.ones(num_agents, dtype=bool)
        self.not_done = np.zeros(num_agents, dtype=bool)
        self.infos = {}

    def observation_space(self, agent):
        type_info = np.iinfo(self._buffers.observations.dtype)
        return gym.spaces.Box(
            low=type_info.min, high=type_info.max,
            shape=(self._buffers.observations.shape[1:]),
            dtype=self._buffers.observations.dtype
        )

    def action_space(self, agent):
        return gym.spaces.MultiDiscrete((self._c_env.num_actions(), 255))

    def get_object_types(self):
        return SimpleNamespace(**{
            k: v["TypeId"] for k, v in self._c_env.get_object_types().items()
        })

    def grid_location_empty(self, r: int, c: int):
        return self._grid[:, r, c].sum() == 0

    def add_object(self, obj_type, r: int, c: int, **props):
        return self._c_env.add_object(obj_type, r, c, **props)

    def _compute_rewards(self):
        '''-1 for each nearby agent'''
        # raw_rewards = 1 - (self.buf.observations==AGENT).sum(axis=(1,2))
        # rewards = np.clip(raw_rewards/10, -1, 0)
        # self.buf.rewards[:] = rewards
        pass

    def render(self):
        raise NotImplementedError

    def reset(self, seed=0):
        self.agents = [i+1 for i in range(self._num_agents)]
        self._c_env.reset(seed)
        return self._buffers.observations, self.infos

    def step(self, actions):
        self.actions = actions
        if use_c:
            self.cenv.step(actions.astype(np.uint32))
        else:
            python_step(self, actions)

        self._compute_rewards()
        self.episode_rewards[self.tick] = self.buf.rewards
        self.tick += 1

        if self.tick >= self.horizon:
            self.agents = []
            self.buf.terminals[:] = self.dones
            self.buf.truncations[:] = self.dones
            infos = {'episode_return': self.episode_rewards.sum(1).mean()}
        else:
            self.buf.terminals[:] = self.not_done
            self.buf.truncations[:] = self.not_done
            infos = self.infos

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, infos)
