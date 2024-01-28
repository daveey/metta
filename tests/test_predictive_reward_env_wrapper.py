import unittest
import gymnasium as gym
import numpy as np
from envs.predictive_reward_env_wrapper import PredictiveRewardEnvWrapper

class TestPredictiveRewardEnvWrapper(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v0')
        self.wrapper = PredictiveRewardEnvWrapper(self.env)

    def test_reset(self):
        obs = self.wrapper.reset()[0]
        self.assertEqual(len(obs), self.env.observation_space.shape[0])

    def test_step(self):
        self.wrapper.reset()
        action = self.env.action_space.sample()
        next_obs_prediction = self.env.observation_space.sample()
        obs, rewards, terminated, truncated, infos = self.wrapper.step((action, next_obs_prediction))
        self.assertEqual(len(obs), self.env.observation_space.shape[0])
        self.assertIsInstance(rewards, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(infos, dict)

    def test_render(self):
        self.wrapper.reset()
        self.wrapper.render()

if __name__ == '__main__':
    unittest.main()
