from __future__ import annotations
from math import inf

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
            act_pred_space = gym.spaces.Tuple([
                act_space,
                gym.spaces.Tuple(
                    [gym.spaces.Discrete(256)] * flat_obs_space.shape[0])
            ])
            spaces.append(act_pred_space)

        self.action_space = MultiAgentActionSpace(spaces)
        self.prediction_error = 0
        self.num_episode_steps = 0

    def reset(self, **kwargs):
        self.prediction_error = 0
        self.num_episode_steps = 0
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, actions):
        (actions, next_obs_prediction) = zip(*actions)
        obs, rewards, terminated, truncated, infos = self.env.step(list(actions))
        step_pred_error = 0
        for i in range(len(obs)):
            clipped_pred = np.clip(next_obs_prediction[i], 0, 255)
            flat_ob = gym.spaces.flatten(self.observation_space[i], obs[i])

            prediction_error = np.sum(np.abs(clipped_pred - flat_ob))
            pred_error_norm = prediction_error / len(flat_ob) / 255.0

            assert pred_error_norm <= 1.0, f"Prediction error {pred_error_norm} is greater than 1.0"
            assert pred_error_norm >= 0, f"Prediction error {pred_error_norm} is less than 0.0"

            rewards[i] = rewards[i] + self.prediction_error_reward * pred_error_norm
            step_pred_error += pred_error_norm

        self.num_episode_steps += 1
        self.prediction_error += step_pred_error / len(obs)

        if terminated or truncated:
            infos["episode_extra_stats"] = {
                "prediction_error_step": self.prediction_error / self.num_episode_steps,
                "prediction_error_episode": self.prediction_error,
            }

        return obs, rewards, terminated, truncated, infos
