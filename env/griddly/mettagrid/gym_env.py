from typing import Any, Dict

import gymnasium as gym
import hydra
from omegaconf import OmegaConf
from env.griddly.griddly_gym_env import GriddlyGymEnv
from env.griddly.mettagrid.game_builder import MettaGridGameBuilder
from env.wrapper.last_action_tracker import LastActionTracker
from env.wrapper.reward_tracker import RewardTracker

class MettaGridGymEnv(gym.Env):
    def __init__(self, render_mode: str, game_builder: MettaGridGameBuilder, **cfg):
        super().__init__()

        self._render_mode = render_mode
        self._game_builder = game_builder
        self._cfg = OmegaConf.create(cfg)

        self.make_env()

    def make_env(self):
        griddly_yaml = self._game_builder.build()
        self._griddly_env = GriddlyGymEnv(
            griddly_yaml,
            self._cfg.max_action_value,
            self._game_builder.obs_width,
            self._game_builder.obs_height,
            self._max_steps,
            self._game_builder.num_agents,
            self._render_mode
        )

        self._env = LastActionTracker(self._griddly_env)
        self._env = RewardTracker(self._env)

    def reset(self, **kwargs):
        self.make_env()
        obs, infos = self._env.reset(**kwargs)
        self._compute_max_energy()
        return obs, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        if terminated or truncated:
            self.process_episode_stats(info["episode_extra_stats"])

        return obs, rewards, terminated, truncated, info

    def process_episode_stats(self, episode_stats: Dict[str, Any]):
        max_energy = self._compute_max_energy()
        max_energy_per_agent = max_energy / self.player_count
        for agent_stats in episode_stats:
            for stat_name in agent_stats.keys():
                if stat_name.startswith("stats_action_"):
                    agent_stats[stat_name] /= self._griddly_env.num_steps

            agent_stats["level_max_energy"] = max_energy
            agent_stats["level_max_energy_per_agent"] = max_energy_per_agent
            agent_stats["level_max_reward_per_agent"] = max_energy_per_agent * 3

    def _compute_max_energy(self):
        return 1
        # # compute the max possible energy for the level
        # charger_energy, generator_cooldown, agent_init, agent_regen = map(
        #     lambda x: float(x[0]),
        #     self._griddly_env.unwrapped.get_global_variables([
        #         "conf:charger:energy",
        #         "conf:generator:cooldown",
        #         "conf:agent:energy:initial",
        #     ).values())

        # num_steps = self._max_steps
        # num_generators = len(list(
        #     filter(lambda x: x["Name"] == "generator",
        #     self._griddly_env.game.get_state()["Objects"])))
        # max_resources = num_generators * (1 + num_steps // generator_cooldown)
        # max_charger_energy = float(charger_energy) * max_resources
        # max_agent_energy = float(agent_init) + agent_regen * num_steps
        # self._max_level_energy = max_charger_energy + self._num_agents * max_agent_energy
        # self._max_level_energy_per_agent = self._max_level_energy / self._num_agents
        # # if the agents use all their energy for the altar, we get a 3x reward
        # self._max_level_reward_per_agent = self._max_level_energy_per_agent * 3


    @property
    def _max_steps(self):
        return self._game_builder.max_steps

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def player_count(self):
        return self._env.unwrapped.player_count
