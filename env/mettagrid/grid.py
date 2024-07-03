from pdb import set_trace as T
import numpy as np

import pettingzoo
import gymnasium

from env.mettagrid import render
import pufferlib

EMPTY = 0
AGENT = 1
WALL = 2

PASS = 0
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4

use_c = True
try:
    from pufferlib.environments.ocean.c_grid import Environment as CEnv
except ImportError:
    use_c = False
    import warnings
    warnings.warn('PufferLib Cython extensions not installed. Using slow Python versions')

def observation_space(env, agent):
    return gymnasium.spaces.Box(
        low=0, high=255, shape=(env.obs_size, env.obs_size), dtype=np.uint8)

def action_space(env, agent):
    return gymnasium.spaces.Discrete(5)

class PufferGrid(pufferlib.PufferEnv):
    def __init__(self,
                 map_size=1024,
                 map_layers=1,
                 num_agents=1024,
                 horizon=1024,
                 vision_range=10,
                 render_mode='rgb_array'):
        super().__init__()
        # api params
        self.emulated = None

        self.map_size = map_size
        self.map_layers = map_layers
        self.num_agents = num_agents
        self.horizon = horizon
        self.vision_range = vision_range
        self.obs_size = 2*self.vision_range + 1

        self.grid = np.zeros((map_layers, map_size, map_size), dtype=np.uint8)
        self.object_positions = np.zeros((num_agents, 2), dtype=np.uint32)
        self.agent_positions = np.zeros((num_agents, 2), dtype=np.uint32)

        channels = map_layers # hit points

        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (num_agents, self.obs_size, self.obs_size, channels), dtype=np.uint8),
            rewards = np.zeros(num_agents, dtype=np.float32),
            terminals = np.zeros(num_agents, dtype=bool),
            truncations = np.zeros(num_agents, dtype=bool),
            masks = np.ones(num_agents, dtype=bool),
        )
        self.actions = np.zeros(num_agents, dtype=np.uint32)
        self.episode_rewards = np.zeros((horizon, num_agents), dtype=np.float32)

        # if use_c:
        #     self.cenv = CEnv(self.grid, self.agent_positions,
        #         self.spawn_position_cands, self.buf.observations,
        #         map_size, num_agents, horizon, vision_range)

        self.dones = np.ones(num_agents, dtype=bool)
        self.not_done = np.zeros(num_agents, dtype=bool)

        self.render_mode = render_mode
        self.renderer = render.make_renderer(map_size, map_size,
            render_mode=render_mode, sprite_sheet_path='./env/mettagrid/puffer-128-sprites.png')

        self.observation_space = observation_space(self, 1)
        self.action_space = action_space(self, 1)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.done = True
        self.infos = {}

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
        return self.renderer.render(self.grid[1, :, :] !=0 ,
            self.agent_positions, self.actions, self.vision_range)

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

class PettingZooGrid(pettingzoo.ParallelEnv):
    def __init__(self, map_size=80, num_agents=3, horizon=1024, vision_range=10, render_mode='human'):
        super().__init__()
        self.env = PufferGrid(map_size, 2, num_agents, horizon, vision_range, render_mode)
        self.possible_agents = [i+1 for i in range(num_agents)]
        self.infos = {i: {} for i in self.possible_agents}
        self.observation_space = observation_space
        self.action_space = action_space

    @property
    def num_agents(self):
        return self.env.num_agents

    def reset(self, seed=0):
        obs, _ = self.env.reset(seed)
        observations = {i+1: obs[i] for i in range(self.num_agents)}
        return observations, self.infos

    def step(self, actions):
        actions = np.array(list(actions.values()), dtype=np.uint32)
        obs, rewards, terminals, truncations, infos = self.env.step(actions)
        observations = {i+1: obs[i] for i in range(self.num_agents)}
        rewards = {i+1: rewards[i] for i in range(self.num_agents)}
        terminals = {i+1: terminals[i] for i in range(self.num_agents)}
        truncations = {i+1: truncations[i] for i in range(self.num_agents)}
        return observations, rewards, terminals, truncations, self.infos

    def render(self):
        return self.env.render()

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

def test_pz_performance(timeout):
    import time
    env = PettingZooGrid()
    actions = [{i+1: e for i, e in enumerate(np.random.randint(0, 5, env.num_agents))}
        for i in range(1000)]
    idx = 0
    dones = {1: True}
    start = time.time()
    while time.time() - start < timeout:
        if all(dones.values()):
            env.reset()
            dones = {1: False}
        else:
            _, _, dones, _, _ = env.step(actions[idx%1000])

        idx += 1

    sps = env.num_agents * idx // timeout
    print(f'PZ SPS: {sps}')

def test_puffer_performance(timeout):
    import time
    env = PufferGrid()
    actions = np.random.randint(0, 5, (1000, env.num_agents))
    idx = 0
    dones = {1: True}
    start = time.time()
    while time.time() - start < timeout:
        if env.done:
            env.reset()
            dones = {1: False}
        else:
            _, _, dones, _, _ = env.step(actions[idx%1000])

        idx += 1

    sps = env.num_agents * idx // timeout
    print(f'Puffer SPS: {sps}')

def test_random_actions(timeout):
    import time
    env = PettingZooGrid()
    actions = [{i+1: e for i, e in enumerate(np.random.randint(0, 5, env.num_agents))}
        for i in range(1000)]
    idx = 0
    dones = {1: True}
    start = time.time()
    while True:
        if all(dones.values()):
            env.reset()
            dones = {1: False}
        else:
            _, _, dones, _, _ = env.step(actions[idx%1000])

        idx += 1
        frame = env.render()
        import cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    test_random_actions(10)
    # test_puffer_performance(10)
    # test_pz_performance(10)
    # use_c = False
    # test_puffer_performance(10)
    # test_pz_performance(10)
