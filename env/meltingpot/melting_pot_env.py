
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
import hashlib
from agent.sprite_encoder import SpriteEncoder

PLAYER_STR_FORMAT = "player_{index}"
SPRITE_SIZE = 8
NUM_GRID_FEATURES = 8

class MeltingPotEnv(pettingzoo_utils.ParallelEnv, gym_utils.EzPickle):
    """An adapter between Melting Pot substrates and PettingZoo's ParallelEnv."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, substrate_name, max_cycles, **cfg):
        env_config = substrate.get_config(substrate_name)
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
        self._sprite_encoder = SpriteEncoder(8, 8, 3, 8, 32)

    def state(self):
        return self._env.observation()

    def reset(self, seed=None):
        """See base class."""
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.last_rewards = {agent: 0 for agent in self.agents}
        self.last_actions = [0 for _ in self.agents]
        return self.timestep_to_observations(timestep), {}

    def step(self, actions):
        self.last_actions = actions
        timestep = self._env.step(actions)
        self.last_rewards = {
            agent: timestep.reward[index].item() for index, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        done = timestep.last() or self.num_cycles >= self.max_cycles
        infos = { agent: {} for agent in self.agents }
        observations = self.timestep_to_observations(timestep)
        return observations, self.last_rewards, done, done, infos

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

    def global_feature_names(self):
        return [
            "ready_to_shoot",
            "collective_reward",
            "interaction_inventories_00",
            "interaction_inventories_01",
            "interaction_inventories_02",
            "interaction_inventories_10",
            "interaction_inventories_11",
            "interaction_inventories_12",
            "inventory_0",
            "inventory_1",
            "inventory_2",
        ]

    def timestep_to_observations(self, timestep: dm_env.TimeStep) -> Mapping[str, Any]:
        gym_observations = {}
        for index, observation in enumerate(timestep.observation):
            player_id = PLAYER_STR_FORMAT.format(index=index)
            gym_observations[player_id] = {
                "grid_obs": self.rgb_to_grid_observation(observation["RGB"]),
                "global_vars": np.concatenate([
                    np.array([observation.get("READY_TO_SHOOT", 0)]),
                    np.array([observation.get("COLLECTIVE_REWARD", 0)]),
                    observation.get("INTERACTION_INVENTORIES", np.zeros(6)).flatten(),
                    observation.get("INVENTORY", np.zeros(3)),
                ]),
                "last_action": np.array([self.last_actions[index], 0]),
                "last_reward": np.array(self.last_rewards[player_id]),
            }
        return gym_observations

    @lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        rgb_space = self.spec_to_space(self._env.observation_spec()[0])["RGB"]
        w = int(rgb_space.shape[0] / SPRITE_SIZE)
        h = int(rgb_space.shape[1] / SPRITE_SIZE)

        agent_obs_space = gym.spaces.Dict({
                "grid_obs": gym.spaces.Box(
                    low=0, high=255,
                    shape=[ NUM_GRID_FEATURES, w, h ],
                    dtype=np.uint8),
                "global_vars": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=[ len(self.global_feature_names()) ],
                    dtype=np.int32),
                "last_action": gym.spaces.Box(
                    low=0, high=255,
                    shape=[2],
                    dtype=np.int32),
                "last_reward": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=[1],
                    dtype=np.float32),
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

    def hash_sprite(self, sprite, embedding_size):
        # Flatten the sprite and convert it to bytes
        sprite_bytes = sprite.flatten().tobytes()

        # Create a hash of the sprite
        sprite_hash = hashlib.sha256(sprite_bytes).hexdigest()

        # Determine the number of characters to take from the hash for each float
        chars_per_float = len(sprite_hash) // embedding_size

        # Initialize an empty list to hold the floats
        sprite_hash_floats = []

        # Convert the hash into floats
        for i in range(embedding_size):
            # Extract the part of the hash for this float
            sprite_hash_part = sprite_hash[i*chars_per_float:(i+1)*chars_per_float]

            # Convert the part of the hash to an integer
            sprite_hash_int = int(sprite_hash_part, 16)

            # Normalize the integer to the range [-1, 1]
            sprite_hash_float = sprite_hash_int / float(2**(4*chars_per_float) - 1) * 2 - 1

            # Append the float to the list of floats
            sprite_hash_floats.append(sprite_hash_float)

        return sprite_hash_floats

    def rgb_to_grid_observation(self, rgb_observation):
        # Initialize an empty list to hold the hashed sprites
        hashed_sprites = []

        # Iterate over the sprites in the observation
        for i in range(0, 88, SPRITE_SIZE):
            for j in range(0, 88, SPRITE_SIZE):
                # Extract the sprite
                sprite = rgb_observation[i:i+SPRITE_SIZE, j:j+SPRITE_SIZE, :]

                # Hash the sprite
                sprite_hash = self.hash_sprite(sprite, NUM_GRID_FEATURES)
                # Append the sprite hash to the list of hashed sprites
                hashed_sprites.append(sprite_hash)

        # Convert the list of hashed sprites to a numpy array
        hashed_sprites = np.array(hashed_sprites)

        # Reshape the array to (11, 11, -1)
        hashed_sprites = hashed_sprites.reshape(11, 11, -1)

        return hashed_sprites.transpose(2, 0, 1)
