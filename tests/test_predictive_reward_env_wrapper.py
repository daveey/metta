import unittest
from unittest.mock import Mock, MagicMock
import gymnasium as gym
import numpy as np
from envs.predictive_reward_env_wrapper import PredictiveRewardEnvWrapper
from griddly.spaces.action_space import MultiAgentActionSpace

class TestPredictiveRewardEnvWrapper(unittest.TestCase):
    def setUp(self):
        self.env = Mock(spec=gym.Env)
        self.env.action_space = MultiAgentActionSpace([gym.spaces.Discrete(2)]*2)
        self.env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.env.step = MagicMock(return_value=(np.array([0]), np.array([0, 1]), False, False, {}))
        self.wrapper = PredictiveRewardEnvWrapper(self.env, prediction_error_reward=0.5)

    def test_step(self):
        actions = [(0, np.array([0.2])), (1, np.array([0.3]))]
        obs, rewards, terminated, truncated, infos = self.wrapper.step(actions)

        # Check that the reward was modified correctly
        np.testing.assert_allclose(rewards, [0 + 0.2 * 0.5, 1 + 0.3 * 0.5])
        self.assertListEqual(infos["true_objectives"].tolist(), [0, 1])

if __name__ == '__main__':
    unittest.main()
