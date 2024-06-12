from functools import lru_cache
import gymnasium as gym
from matplotlib.pylab import f
import numpy as np

class FeatureMasker(gym.Wrapper):
    def __init__(self, env, masked_features):
        super().__init__(env)
        self._masked_grid_obs = [
            self.env.unwrapped.grid_features.index(feature)
            for feature in masked_features.grid_obs
        ]
        self._self_pos = (
            self.env.unwrapped._obs_width // 2,
            self.env.unwrapped._obs_height // 2)

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        return self._augment_observations(obs), infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return self._augment_observations(obs), rewards, terms, truncs, infos

    def _augment_observations(self, obs):
        if len(self._masked_grid_obs):
            for agent_obs in obs:
                self_obs = agent_obs["grid_obs"][self._masked_grid_obs, *self._self_pos]
                agent_obs["grid_obs"][self._masked_grid_obs] = 0
                agent_obs["grid_obs"][self._masked_grid_obs, *self._self_pos] = self_obs
        return obs
