
from functools import lru_cache

from gymnasium import utils as gym_utils
import matplotlib.pyplot as plt
from meltingpot import substrate
from ml_collections import config_dict
from pettingzoo import utils as pettingzoo_utils
from typing import Any, Mapping

import dm_env
from gymnasium import spaces
import numpy as np
import tree
import gymnasium as gym

PLAYER_STR_FORMAT = "player_{index}"
_WORLD_PREFIX = "WORLD."


class MeltingPotEnv(pettingzoo_utils.ParallelEnv, gym_utils.EzPickle):
    """An adapter between Melting Pot substrates and PettingZoo's ParallelEnv."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_config, max_cycles):
        gym_utils.EzPickle.__init__(self, env_config, max_cycles)

        self.env_config = config_dict.ConfigDict(env_config)
        self.max_cycles = max_cycles
        self._env = substrate.build_from_config(
            self.env_config, roles=self.env_config.default_player_roles
        )
        self._num_players = len(self._env.observation_spec())
        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self._num_players)
        ]

        # self.state_space = self.spec_to_space(self._env.observation_spec()[0]["WORLD.RGB"])

    def state(self):
        return self._env.observation()

    def reset(self, seed=None):
        """See base class."""
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        return self.timestep_to_observations(timestep), {}

    def step(self, actions):
        timestep = self._env.step(actions)
        rewards = {
            agent: timestep.reward[index] for index, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        done = timestep.last() or self.num_cycles >= self.max_cycles
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if done:
            self.agents = []

        observations = self.timestep_to_observations(timestep)
        return observations, rewards, dones, dones, infos

    def close(self):
        """See base class."""
        self._env.close()

    def render(self, mode="human", filename=None):
        rgb_arr = self.state()["WORLD.RGB"]
        if mode == "human":
            plt.cla()
            plt.imshow(rgb_arr, interpolation="nearest")
            if filename is None:
                plt.show(block=False)
            else:
                plt.savefig(filename)
            return None
        return rgb_arr

    @property
    def player_count(self):
        return self._num_players

    def grid_feature_names(self):
        return ["r", "g", "b"]

    def global_feature_names(self):
        return [
            "ready_to_shoot",
            "collective_reward",
            "last_action",
            "last_reward"]

    def timestep_to_observations(self, timestep: dm_env.TimeStep) -> Mapping[str, Any]:
        gym_observations = {}
        for index, observation in enumerate(timestep.observation):
            gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
                "grid_obs": observation["RGB"],
                "global_vars": np.array([
                    observation["READY_TO_SHOOT"],
                    observation["COLLECTIVE_REWARD"],
                    0, # last_action
                    0.0, # last_reward
                ]),
            }
        return gym_observations

    @lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        obs_space = self.spec_to_space(self._env.observation_spec()[0])

        agent_obs_space = gym.spaces.Dict({
                "grid_obs_rgb": obs_space["RGB"],
                "global_vars": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=[ len(self.global_feature_names()) ],
                    dtype=np.int32),
            })
        return agent_obs_space

    @lru_cache(maxsize=None)
    def action_space(self, agent_id):
        return self.spec_to_space(self._env.action_spec()[0])

    def spec_to_space(self, spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
        """Converts a dm_env nested structure of specs to a Gym Space.

        BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
        Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

        Args:
        spec: The nested structure of specs

        Returns:
        The Gym space corresponding to the given spec.
        """
        if isinstance(spec, dm_env.specs.DiscreteArray):
            return spaces.Discrete(spec.num_values)
        elif isinstance(spec, dm_env.specs.BoundedArray):
            return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
        elif isinstance(spec, dm_env.specs.Array):
            if np.issubdtype(spec.dtype, np.floating):
                return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
            elif np.issubdtype(spec.dtype, np.integer):
                info = np.iinfo(spec.dtype)
                return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
            else:
                raise NotImplementedError(f"Unsupported dtype {spec.dtype}")
        elif isinstance(spec, (list, tuple)):
            return spaces.Tuple([self.spec_to_space(s) for s in spec])
        elif isinstance(spec, dict):
            return spaces.Dict({key: self.spec_to_space(s) for key, s in spec.items()})
        else:
            raise ValueError("Unexpected spec of type {}: {}".format(type(spec), spec))
