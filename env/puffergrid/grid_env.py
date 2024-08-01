from pdb import set_trace as T
from types import SimpleNamespace
from typing import Dict, List
import numpy as np

import pettingzoo
import gymnasium as gym

from env.mettagrid import render
import pufferlib
from pufferlib.environment import PufferEnv

class PufferGridEnv(PufferEnv):
    def __init__(
            self,
            c_env,
            num_agents=1,
            max_timesteps=1000,
            obs_width=11,
            obs_height=11) -> None:

        super().__init__()
        self._map_width = c_env.map_width()
        self._map_height = c_env.map_height()
        self._num_agents = num_agents
        self._obs_width = obs_width
        self._obs_height = obs_height
        self._max_timesteps = max_timesteps

        self._c_env = c_env

        self._agent_ids_list = []
        self._agent_ids = np.zeros(num_agents, dtype=np.uint32)
        self.type_ids = SimpleNamespace(**self._c_env.type_ids())
        self.object_dtypes = SimpleNamespace(**self._c_env.dtypes())
        self._num_features = self._c_env.num_features()

        self._grid = np.asarray(self._c_env.grid())

        self._episode_rewards = np.zeros(num_agents, dtype=np.float32)
        self._buffers = self._make_buffers()

    @property
    def observation_space(self):
        type_info = np.iinfo(self._buffers.observations.dtype)
        return gym.spaces.Dict({
            "grid_obs": gym.spaces.Box(
                low=type_info.min, high=type_info.max,
                shape=(self._buffers.observations.shape[1:]),
                dtype=self._buffers.observations.dtype
            ),
            "global_vars": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=[ 0 ],
                dtype=np.int32
            ),
        })

    @property
    def action_space(self):
        return gym.spaces.MultiDiscrete((self._c_env.num_actions(), 255))

    def _make_buffers(self):
        return SimpleNamespace(
            observations=np.zeros((self._num_agents, self._num_features, self._obs_height,  self._obs_width), dtype=np.uint32),
            actions=np.zeros((self._num_agents, 2), dtype=np.uint32),
            rewards=np.zeros(self._num_agents, dtype=np.float32),
            terminals=np.zeros(self._num_agents, dtype=bool),
            truncations=np.zeros(self._num_agents, dtype=bool),
        )

    def grid_location_empty(self, r: int, c: int):
        return sum(self._grid[r, c]) == 0

    def add_agent(self, type_id: int, r: int, c: int, layer: int, **props) -> int:
        id = self._c_env.add_object(type_id, r, c, layer, **props)
        assert id >= 0, "Failed to add object"
        self._agent_ids_list.append(id)
        self._agent_ids[len(self._agent_ids_list)-1] = id
        return id

    def add_object(self, type_id: int, r: int, c: int, layer: int, **props) -> int:
        id =  self._c_env.add_object(type_id, r, c, layer, **props)
        assert id >= 0, "Failed to add object"

    def render(self):
        raise NotImplementedError

    def reset(self, seed=0):
        assert self._c_env.current_timestep() == 0, "Reset not supported"
        obs = self._compute_observations()
        return obs, {}

    def step(self, actions):
        # print("actions", self._agent_ids, actions)
        # actions = np.array(actions, dtype=np.uint32)
        self._c_env.step(
            self._agent_ids, np.array(actions, dtype=np.uint32),
            self._buffers.rewards, self._buffers.terminals)

        self._episode_rewards += self._buffers.rewards
        obs = self._compute_observations()

        infos = {}
        # if self.current_timestep >= self._max_timesteps:
        #     self._buffers.terminals.fill(True)
        #     self._buffers.truncations.fill(True)
        #     infos = {
        #         "episode_return": self._episode_rewards.mean(),
        #         "episode_length": self.current_timestep,
        #         "episode_stats": self._c_env.get_episode_stats()
        #     }

        return (obs,
                self._buffers.rewards,
                self._buffers.terminals,
                self._buffers.truncations,
                infos)

    def _compute_observations(self):
        # self._buffers.observations.fill(0)
        self._c_env.compute_observations(
            self._agent_ids,
            self._obs_width, self._obs_height,
            self._buffers.observations)

        # obs = []
        # for agent in range(self._num_agents):
        #     obs.append({
        #         "grid_obs": self._buffers.observations[agent],
        #         "global_vars": np.zeros(0, dtype=np.uint32)
        #     })
        # return obs
        return {}

    @property
    def current_timestep(self):
        return self._c_env.current_timestep()

    @property
    def unwrapped(self):
        return self

    @property
    def player_count(self):
        return self._num_agents

    @property
    def grid_features(self):
        return self._c_env.grid_features()

    @property
    def global_features(self):
        return self._c_env.global_features()
