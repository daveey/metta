from collections import defaultdict
from typing import Optional

import gymnasium as gym
from typing import Any, Dict, Optional

import numpy as np

from griddly import GymWrapperFactory
from griddly.wrappers.render_wrapper import RenderWrapper
from griddly.gym import GymWrapper
from griddly import gd
from griddly.util.render_tools import RenderToVideo

from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface

class GridmanMultiEnv(gym.Env, TrainingInfoInterface):

    def __init__(self, full_env_name, cfg, render_mode: Optional[str] = None):
        TrainingInfoInterface.__init__(self)

        self.name = full_env_name
        self.cfg = cfg

        self.gym_env = GymWrapper(
            "./envs/griddly/gridman/gridman_multiagent.yaml",
            player_observer_type="VectorGridMan",
            global_observer_type="HumanPlayerBlockObserver",
            level=0,
            max_steps=2000,
            render_mode=render_mode
        )
        self.gym_env_global = RenderWrapper(self.gym_env, "global")
        self.num_agents = self.gym_env.player_count

        self._player_done_variable = "done_variable"
        self._active_agents = set()

        self.curr_episode_steps = 0
        self.observation_space = self.gym_env.observation_space[0]
        self.action_space = self.gym_env.action_space[0]
        self.is_multiagent = True
        self.inactive_steps = [3] * self.num_agents
        self.episode_rewards = [[] for _ in range(self.num_agents)]

        self.render_mode = render_mode
        self.global_recorder = RenderToVideo(self.gym_env_global, "global_video_test.mp4")

    def reset(self, **kwargs):
        self._active_agents.update([a + 1 for a in range(self.num_agents)])
        self.curr_episode_steps = 0
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        return self.gym_env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.gym_env.step(actions)
        self.curr_episode_steps += 1

        if self._player_done_variable is not None:
            griddly_players_done = self._resolve_player_done_variable()
            assert len(griddly_players_done) == 0

        # auto-reset the environment
        if terminated or truncated:
            obs = self.gym_env.reset()[0]

        #     for agent_id in self._active_agents:
        #         done_map[agent_id] = griddly_players_done[agent_id] == 1 or all_done
        # else:
        #     for p in range(self.num_agents):
        #         done_map[p] = False

        # Finally remove any agent ids that are done
        # for agent_id, is_done in done_map.items():
        #     if is_done:
        #         self._active_agents.discard(agent_id)

        terminated = [terminated] * self.num_agents
        truncated = [truncated] * self.num_agents
        infos = [{}] * self.num_agents
        return obs, rewards, terminated, truncated, infos

    def render(self, *args, **kwargs):
        self.gym_env_global.render()
        self.global_recorder.capture_frame()

    def _resolve_player_done_variable(self):
        resolved_variables = self.gym_env.game.get_global_variable([self._player_done_variable])
        return resolved_variables[self._player_done_variable]
