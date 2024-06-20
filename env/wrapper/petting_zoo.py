import pettingzoo
import gymnasium as gym
import numpy as np

class PettingZooEnvWrapper(pettingzoo.ParallelEnv):
    def __init__(self, gym_env: gym.Env):
        super().__init__()
        self._gym_env = gym_env
        self.possible_agents = [i+1 for i in range(self.num_agents)]
        # agents gets manipulated
        self.agents = [i+1 for i in range(self.num_agents)]
        self.infos = {i: {} for i in self.possible_agents}

    @property
    def num_agents(self):
        return self._gym_env.unwrapped.player_count

    def observation_space(self, agent: int) -> gym.Space:
        return self._gym_env.observation_space

    def action_space(self, agent: int) -> gym.Space:
        return self._gym_env.action_space

    def reset(self, seed=0):
        obs, _ = self._gym_env.reset()
        observations = {i+1: obs[i] for i in range(self.num_agents)}
        return observations, self.infos

    def step(self, actions):
        actions = np.array(list(actions.values()), dtype=np.uint32)
        obs, rewards, terminated, truncated, infos = self._gym_env.step(actions)
        observations = {i+1: obs[i] for i in range(self.num_agents)}
        rewards = {i+1: rewards[i] for i in range(self.num_agents)}
        terminated = {i+1: terminated for i in range(self.num_agents)}
        truncated = {i+1: truncated for i in range(self.num_agents)}
        return observations, rewards, terminated, truncated, self.infos
