from ast import arg
from collections import defaultdict
from typing import Callable, Optional

import gymnasium as gym
from typing import Any, Dict, Optional

import numpy as np

from griddly import GymWrapperFactory
from griddly.wrappers.render_wrapper import RenderWrapper
from griddly.gym import GymWrapper
from griddly import gd
from griddly.util.render_tools import RenderToVideo

from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface

class GriddlyEnvWrapper(gym.Env, TrainingInfoInterface):

    def __init__(self,
                 griddly_env: gym.Env,
                 render_mode: Optional[str] = None,
                 make_level: Optional[Callable] = None
                 ):
        TrainingInfoInterface.__init__(self)

        # self.name = full_env_name
        # self.cfg = cfg

        self.gym_env = griddly_env
        self.gym_env_global = RenderWrapper(self.gym_env, "global", render_mode=render_mode)
        self.num_agents = self.gym_env.player_count
        self.make_level = make_level

        self.curr_episode_steps = 0
        self.observation_space = self.gym_env.observation_space[0]
        action_space = self.gym_env.action_space[0]
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            action_space = gym.spaces.Tuple(
                [gym.spaces.Discrete(num_actions) for num_actions in action_space.nvec]
            )

        self.action_space = action_space
        self.is_multiagent = True
        self.episode_rewards = [[] for _ in range(self.num_agents)]

    def reset(self, **kwargs):
        self.curr_episode_steps = 0
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        if self.make_level is not None:
            level_string = self.make_level()
            return self.gym_env.reset(options={"level_string": level_string})
        else:
            return self.gym_env.reset()

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.gym_env.step(list(actions))
        self.curr_episode_steps += 1

        # auto-reset the environment
        if terminated or truncated:
            obs = self.reset()[0]

        terminated = [terminated] * self.num_agents
        truncated = [truncated] * self.num_agents

        tos = None
        if "true_objectives" in infos:
            tos = infos["true_objectives"]
            del infos["true_objectives"]

        infos = [infos] * self.num_agents
        if tos is not None:
            for i in range(self.num_agents):
                infos[i]["true_objectives"] = tos[i]

        return obs, rewards, terminated, truncated, infos

    def render(self, *args, **kwargs):
        return self.gym_env_global.render()
