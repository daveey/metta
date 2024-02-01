from argparse import Namespace
from types import SimpleNamespace
import unittest
import gymnasium as gym
import numpy as np
from envs.griddly.forage.env import ForageEnvFactory
from envs.predictive_reward_env_wrapper import PredictiveRewardEnvWrapper

class TestPredictiveForageEnv(unittest.TestCase):
    def setUp(self):
        cfg = SimpleNamespace(**{
            "forage_num_agents": 3,
            "forage_max_env_steps": 100,
            "forage_width_min": 9,
            "forage_width_max": 10,
            "forage_height_min": 9,
            "forage_height_max": 10,
            "forage_wall_density": 0,
            "forage_energy_per_agent": 1,
            "forage_prediction_error_reward": 10,
        })
        self.factory = ForageEnvFactory(cfg)
        self.env = self.factory.make()

    def test_step(self):
        obs = self.env.reset()[0]
        actions = zip(
            [0, 0, 0], # actions
            [np.array([e]) for e in [0.02, 0.1, 0]]) # prediction errors

        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self.assertEqual(rewards, [0.2, 1.0, 0.0])
        self.assertEqual(infos["true_objectives"], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
