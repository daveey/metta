from pdb import set_trace as T
from types import SimpleNamespace
from typing import List
import numpy as np

import pettingzoo
import gymnasium

from env.mettagrid import render
import pufferlib
from pufferlib.environment import PufferEnv

class PufferGridEnv(PufferEnv):
    def __init__(
            self,
            c_env_class,
            map_width=50,
            map_height=50,
            num_agents=1,
            max_timesteps=1000,
            obs_width=11,
            obs_height=11) -> None:

        super().__init__()
        self._map_width = map_width
        self._map_height = map_height
        self._num_agents = num_agents
        self._obs_width = obs_width
        self._obs_height = obs_height

        self._c_env = c_env_class(
            map_width,
            map_height,
            num_agents,
            max_timesteps,
            obs_width,
            obs_height)

        self._buffers = SimpleNamespace(**{
            k: np.asarray(v) for k,v in self._c_env.get_buffers().items()
        })
        self._grid = np.asarray(self._c_env.get_grid())

        self.actions = np.zeros(num_agents, dtype=np.uint32)
        self.episode_rewards = np.zeros(num_agents, dtype=np.float32)
        self.dones = np.ones(num_agents, dtype=bool)
        self.not_done = np.zeros(num_agents, dtype=bool)
        self.done = True
        self.infos = {}



    # def observation_space(env, agent):
    #     return gymnasium.spaces.Box(
    #         low=0, high=255, shape=(env.obs_size, env.obs_size), dtype=np.uint8)

    def action_space(self, agent):
        return gymnasium.spaces.MultiDiscrete(self._c_env.num_actions(), 255)

    def _compute_observations(self):
        for agent_idx in range(self.num_agents):
            r = self.agent_positions[agent_idx, 0]
            c = self.agent_positions[agent_idx, 1]
            for layer in range(self.grid.shape[0]):
                self.buf.observations[agent_idx, :, :, layer] = self.grid[
                    layer,
                    r-self.vision_range:r+self.vision_range+1,
                    c-self.vision_range:c+self.vision_range+1,
                ]

    def _compute_rewards(self):
        '''-1 for each nearby agent'''
        # raw_rewards = 1 - (self.buf.observations==AGENT).sum(axis=(1,2))
        # rewards = np.clip(raw_rewards/10, -1, 0)
        # self.buf.rewards[:] = rewards
        pass

    def render(self):
        raise NotImplementedError

    def reset(self, seed=0):
        self.agents = [i+1 for i in range(self.num_agents)]
        self.done = False
        self.tick = 1

        self.grid.fill(0)
        self.episode_rewards.fill(0)
        if use_c:
            self.cenv.reset(self.buf.observations, seed)
        else:
            python_reset(self)

        return self.buf.observations, self.infos

    def step(self, actions):
        self.actions = actions
        if use_c:
            self.cenv.step(actions.astype(np.uint32))
        else:
            python_step(self, actions)

        self._compute_rewards()
        self.episode_rewards[self.tick] = self.buf.rewards
        self.tick += 1

        if self.tick >= self.horizon:
            self.done = True
            self.agents = []
            self.buf.terminals[:] = self.dones
            self.buf.truncations[:] = self.dones
            infos = {'episode_return': self.episode_rewards.sum(1).mean()}
        else:
            self.buf.terminals[:] = self.not_done
            self.buf.truncations[:] = self.not_done
            infos = self.infos

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, infos)

def python_reset(self):
    # Add borders
    left = self.vision_range
    right = self.map_size - self.vision_range - 1
    self.grid[0, :left, :] = WALL
    self.grid[0, right:, :] = WALL
    self.grid[0, :, :left] = WALL
    self.grid[0, :, right:] = WALL

    # Agent spawning
    for agent_idx in range(self.num_agents):
        r = 20 + agent_idx * 2
        c = 20 + agent_idx * 2
        if self.grid[0, r, c] == 0:
            self.grid[1, r, c] = AGENT
            self.agent_positions[agent_idx, 0] = r
            self.agent_positions[agent_idx, 1] = c
            agent_idx += 1
            if agent_idx == self.num_agents:
                break

    self._compute_observations()

def python_step(self, actions):
    for agent_idx in range(self.num_agents):
        r = self.agent_positions[agent_idx, 0]
        c = self.agent_positions[agent_idx, 1]
        atn = actions[agent_idx]
        dr = 0
        dc = 0
        if atn == PASS:
            continue
        elif atn == NORTH:
            dr = -1
        elif atn == SOUTH:
            dr = 1
        elif atn == EAST:
            dc = 1
        elif atn == WEST:
            dc = -1
        else:
            raise ValueError(f'Invalid action: {atn}')

        dest_r = r + dr
        dest_c = c + dc

        if self.grid[1, dest_r, dest_c] == 0:
            self.grid[1, r, c] = EMPTY
            self.grid[1, dest_r, dest_c] = AGENT
            self.agent_positions[agent_idx, 0] = dest_r
            self.agent_positions[agent_idx, 1] = dest_c

    self._compute_observations()
