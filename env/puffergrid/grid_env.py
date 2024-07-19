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
            map_width = map_width,
            map_height = map_height,
            max_timesteps = max_timesteps)

        self._agent_ids_list = []
        self._agent_ids = np.zeros(num_agents, dtype=np.uint32)
        self.type_ids = SimpleNamespace(**self._c_env.get_types())
        self.object_dtypes = SimpleNamespace(**self._c_env.get_dtypes())
        self._num_features = self._c_env.get_num_features()

        self._grid = np.asarray(self._c_env.get_grid())

        self.episode_rewards = np.zeros(num_agents, dtype=np.float32)
        self.dones = np.ones(num_agents, dtype=bool)
        self.not_done = np.zeros(num_agents, dtype=bool)
        self.infos = {}
        self._buffers = self._make_buffers()

    def observation_space(self, agent):
        type_info = np.iinfo(self._buffers.observations.dtype)
        return gym.spaces.Box(
            low=type_info.min, high=type_info.max,
            shape=(self._buffers.observations.shape[1:]),
            dtype=self._buffers.observations.dtype
        )

    def action_space(self, agent):
        return gym.spaces.MultiDiscrete((self._c_env.num_actions(), 255))

    def _make_buffers(self):
        return SimpleNamespace(
            observations=np.zeros((self._num_agents, self._obs_width, self._obs_height, self._num_features), dtype=np.uint32),
            actions=np.zeros((self._num_agents, 2), dtype=np.uint32),
            rewards=np.zeros(self._num_agents, dtype=np.float32),
            dones=np.zeros(self._num_agents, dtype=bool),
        )

    def grid_location_empty(self, r: int, c: int):
        return sum(self._grid[r, c]) == 0

    def add_agent(self, obj_type, r: int, c: int, **props) -> int:
        id = self._c_env.add_object(obj_type, r, c, **props)
        self._agent_ids_list.append(id)
        self._agent_ids[len(self._agent_ids_list)-1] = id
        return id

    def add_object(self, obj_type, r: int, c: int, **props) -> int:
        return self._c_env.add_object(obj_type, r, c, **props)

    def render(self):
        raise NotImplementedError

    def reset(self, seed=0):
        assert self._c_env.get_current_timestep() == 0, "Reset not supported"
        self._compute_observations()
        return self._buffers.observations, self.infos

    def step(self, actions):
        # print("actions", self._agent_ids, actions)
        self._buffers.rewards.fill(0)
        self._buffers.dones.fill(False)

        self._c_env.step(
            self._agent_ids, actions,
            self._buffers.rewards, self._buffers.dones)

        self._compute_observations()

        # if self.tick >= self.horizon:
        #     self.agents = []
        #     self.buf.terminals[:] = self.dones
        #     self.buf.truncations[:] = self.dones
        #     infos = {'episode_return': self.episode_rewards.sum(1).mean()}
        # else:
        #     self.buf.terminals[:] = self.not_done
        #     self.buf.truncations[:] = self.not_done
        #     infos = self.infos

        return (self._buffers.observations,
                self._buffers.rewards,
                self._buffers.dones,
                self._buffers.dones,
                self.infos)

    def _compute_observations(self):
        self._buffers.observations.fill(0)
        self._c_env.compute_observations(
            self._agent_ids,
            self._obs_width, self._obs_height,
            self._buffers.observations)

    @property
    def current_timestep(self):
        return self._c_env.get_current_timestep()
