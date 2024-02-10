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
from griddly.util.render_tools import RenderToVideo, RenderToFile

from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface


class GriddlyEnvWrapper(gym.Env, TrainingInfoInterface):

    def __init__(
        self,
        griddly_env: gym.Env,
        render_mode: Optional[str] = None,
        make_level: Optional[Callable] = None,
    ):
        TrainingInfoInterface.__init__(self)

        # self.name = full_env_name
        # self.cfg = cfg

        self.gym_env = griddly_env
        self.gym_env_global = RenderWrapper(
            self.gym_env, "global", render_mode=render_mode
        )
        self.num_agents = self.gym_env.player_count
        self.make_level = make_level

        self.curr_episode_steps = 0
        if self.num_agents == 1:
            self.observation_space = self.gym_env.observation_space
            action_space = self.gym_env.action_space
            self.is_multiagent = False
        else:
            self.observation_space = self.gym_env.observation_space[0]
            action_space = self.gym_env.action_space[0]
            self.is_multiagent = True

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            action_space = gym.spaces.Tuple(
                [gym.spaces.Discrete(num_actions) for num_actions in action_space.nvec]
            )

        self.action_space = action_space
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        self.dump_state = False

    def _dump_global_obs(self, label: str):
        if self.dump_state:
            r = RenderToFile()
            r.render(self.gym_env_global.render(),
                     f"/tmp/sample_factory_state/{self.curr_episode_steps}.{label}.png")

    def reset(self, **kwargs):
        self.curr_episode_steps = 0
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        if self.make_level is not None:
            level_string = self.make_level()
            return self.gym_env.reset(options={"level_string": level_string})
        else:
            return self.gym_env.reset()

    def step(self, actions):
        if self.is_multiagent:
            actions = list(actions)

        self._dump_global_obs("pre_step")
        obs, rewards, terminated, truncated, infos_dict = self.gym_env.step(actions)
        self._dump_global_obs("post_step")
        self.curr_episode_steps += 1

        # auto-reset the environment
        if terminated or truncated:
            obs = self.reset()[0]

        # For better readability, make `infos` a list.
        # In case of a single player, get the first element before returning
        infos = [infos_dict.copy()] * self.num_agents
        if "episode_extra_stats" in infos_dict:
            for i, info in enumerate(infos):
                info["episode_extra_stats"] = infos_dict["episode_extra_stats"][i]

        if self.is_multiagent:
            terminated = [terminated] * self.num_agents
            truncated = [truncated] * self.num_agents
        else:
            infos = infos[0]

        return obs, rewards, terminated, truncated, infos

    def render(self, *args, **kwargs):
        return self.gym_env_global.render()
