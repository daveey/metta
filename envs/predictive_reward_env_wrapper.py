from __future__ import annotations
from math import inf

import gymnasium as gym

import numpy as np

from copy import deepcopy
from typing import TYPE_CHECKING
from sample_factory.utils.timing import Timing
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

        # add total prediction error to the action_space of each agent
        self.action_space = MultiAgentActionSpace([
            gym.spaces.Tuple([
                agent_action_space,
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)]
            )
            for agent_action_space in action_space
        ])
        self.num_episode_steps = 0
        self.episode_prediction_error = 0

    def reset(self, **kwargs):
        self.episode_prediction_error = 0
        self.num_episode_steps = 0
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, actions):
        (actions, prediction_error) = zip(*actions)

        obs, rewards, terminated, truncated, infos = self.env.step(list(actions))

        infos["true_objectives"] = deepcopy(rewards)
        rewards = np.array(rewards, dtype=np.float32)
        prediction_error = np.array(prediction_error).flatten()
        rewards += prediction_error * self.prediction_error_reward
        step_pred_error = np.mean(prediction_error)

        # for i in range(len(obs)):
            # pe = prediction_error[i].item()
            # rewards[i] += pe * self.prediction_error_reward
            # step_pred_error += pe

        self.num_episode_steps += 1
        self.episode_prediction_error += step_pred_error / len(obs)

        if terminated or truncated:
            infos["episode_extra_stats"] = {
                "prediction_error_step": self.episode_prediction_error / self.num_episode_steps,
                "prediction_error_episode": self.episode_prediction_error,
            }

        return obs, rewards, terminated, truncated, infos
