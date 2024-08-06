from pdb import set_trace as T
from types import SimpleNamespace
import numpy as np

import pettingzoo
import gymnasium as gym

from env.griddly.builder import action
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
        self._num_features = len(self.grid_features)

        # self._grid = np.asarray(self._c_env.grid())

        self._episode_rewards = np.zeros(num_agents, dtype=np.float32)
        self._buffers = self._make_buffers()

    @property
    def observation_space(self):
        type_info = np.iinfo(self._buffers.observations.dtype)
        # return gym.spaces.Dict({
        #     "grid_obs": gym.spaces.Box(
        #         low=type_info.min, high=type_info.max,
        #         shape=(self._buffers.observations.shape[1:]),
        #         dtype=self._buffers.observations.dtype
        #     ),
        #     "global_vars": gym.spaces.Box(
        #         low=-np.inf, high=np.inf,
        #         shape=[ 0 ],
        #         dtype=np.int32
        #     ),
        return gym.spaces.Box(
            low=type_info.min, high=type_info.max,
            shape=(self._buffers.observations.shape[1:]),
            dtype=self._buffers.observations.dtype
        )


    @property
    def action_space(self):
        return gym.spaces.MultiDiscrete((self._c_env.num_actions(), 255))

    def _make_buffers(self):
        return SimpleNamespace(
            observations=np.zeros((self._num_agents, self._num_features, self._obs_height,  self._obs_width), dtype=np.int32),
            rewards=np.zeros(self._num_agents, dtype=np.float32),
            terminals=np.zeros(self._num_agents, dtype=bool),
            truncations=np.zeros(self._num_agents, dtype=bool),
            masks=np.ones(self._num_agents, dtype=bool),
        )
    def set_buffers(self, buffers):
        self._buffers = buffers


    def render(self):
        raise NotImplementedError

    def reset(self, seed=0):
        assert self._c_env.current_timestep() == 0, "Reset not supported"

        self._c_env.set_buffers(
            self._buffers.observations,
            self._buffers.terminals,
            self._buffers.rewards)

        self._buffers.observations.fill(0)
        self._c_env.reset()
        return self._buffers.observations, {}

    def step(self, actions):
        self._buffers.observations.fill(0)
        self._c_env.step(actions.astype(np.uint32))

        self._episode_rewards += self._buffers.rewards

        infos = {}
        if self.current_timestep >= self._max_timesteps:
            self._buffers.terminals.fill(True)
            self._buffers.truncations.fill(True)
            infos = {
                "episode_return": self._episode_rewards.mean(),
                "episode_length": self.current_timestep,
                "episode_stats": self._c_env.get_episode_stats()
            }
        return (self._buffers.observations,
                self._buffers.rewards,
                self._buffers.terminals,
                self._buffers.truncations,
                infos)

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
