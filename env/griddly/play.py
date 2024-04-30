import re
import sys

import gymnasium

from sample_factory.enjoy import enjoy

from gymnasium.utils.play import play
from griddly import GymWrapperFactory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from rl_framework.sample_factory.sample_factory_env_wrapper import SampleFactoryEnvWrapper
from agent import agent
import numpy as np
from env.griddly.power_grid import power_grid_env, power_grid_level_generator
from griddly.wrappers.render_wrapper import RenderWrapper


def main():
    """Script entry point."""
    cfg = train.parse_custom_args(evaluation=True)
    lg = power_grid_level_generator.PowerGridLevelGenerator(cfg)
    env = power_grid_env.PowerGridEnvWrapper.make_env(cfg, level_generator=lg)
    env.enable_history(True)
    genv =  RenderWrapper(env, "global", render_mode="rgb_array")
    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        # print(info)
        pass

    actions = [np.array([0, 0])] * (cfg.env_num_agents - 1)
    play(genv, fps=5, keys_to_action={
        "w": [np.array([0, 2])] + actions,
        "s": [np.array([0, 4])] + actions,
        "d": [np.array([0, 3])] + actions,
        "a": [np.array([0, 1])] + actions,
        "e": [np.array([1, 0])] + actions,
       },
       noop=[np.array([0,0])] * cfg.env_num_agents,
       callback=callback)
if __name__ == "__main__":
    main()
