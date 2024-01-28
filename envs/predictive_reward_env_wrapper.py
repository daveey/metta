from __future__ import annotations

import gymnasium as gym

import numpy as np

from copy import deepcopy
from typing import TYPE_CHECKING

import gymnasium as gym
from griddly.spaces.action_space import MultiAgentActionSpace

class PredictiveRewardEnvWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, prediction_error_reward: float = 0):
        self.prediction_error_reward = prediction_error_reward
        gym.utils.RecordConstructorArgs.__init__(self)

        gym.Wrapper.__init__(self, env)

        self.observation_space = self.env.observation_space
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            action_space = gym.spaces.Tuple(
                [gym.spaces.Discrete(num_actions) for num_actions in action_space.nvec]
            )

        spaces = []
        for act_space, obs_space in zip(action_space, self.observation_space):
            flat_obs_space = gym.spaces.flatten_space(obs_space)
            act_pred_space = gym.spaces.Tuple([act_space, flat_obs_space])
            spaces.append(act_pred_space)

        self.action_space = MultiAgentActionSpace(spaces)

    def step(self, actions):
        (actions, next_obs_prediction) = zip(*actions)
        obs, rewards, terminated, truncated, infos = self.env.step(list(actions))
        for i in range(len(obs)):
            flat_ob = gym.spaces.flatten(self.observation_space[i], obs[i])
            prediction_error = np.sum(np.abs(next_obs_prediction[i] - flat_ob))
            pred_error_norm = prediction_error / len(flat_ob) / 255.0
            rewards[i] = rewards[i] + self.prediction_error_reward * pred_error_norm
            infos.setdefault(i, {})["prediction_error"] = prediction_error

        return obs, rewards, terminated, truncated, infos
