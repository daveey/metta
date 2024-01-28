import unittest
import gymnasium as gym
import numpy as np
from envs.griddly.forage.env import ForageEnvFactory
from envs.predictive_reward_env_wrapper import PredictiveRewardEnvWrapper

class TestPredictiveForageEnv(unittest.TestCase):
    def setUp(self):
        cfg = {
            "forage.num_agents": 3,
            "forage.max_env_steps": 100,
            "forage.width_min": 9,
            "forage.width_max": 10,
            "forage.height_min": 9,
            "forage.height_max": 10,
            "forage.wall_density": 0,
            "forage.energy_per_agent": 1,
            "forage.prediction_error_reward": 10,
        }
        self.factory = ForageEnvFactory(cfg)
        self.env = self.factory.make()

    def test_step(self):
        obs = self.env.reset()[0]
        actions = zip([0, 0, 0], obs)
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self.assertEqual(rewards, [0.0, 0.0, 0.0])

        # move A3 around, his error should be > than the other agents
        total_rewards = 0
        for dir in range(1, 5):
            actions = zip([0, 0, dir], obs)
            obs, rewards, terminated, truncated, infos = self.env.step(actions)
            self.assertLessEqual(rewards[0], rewards[2])
            self.assertLessEqual(rewards[1], rewards[2])
            total_rewards += sum(rewards)

        self.assertGreater(total_rewards, 0.0)

if __name__ == '__main__':
    unittest.main()
