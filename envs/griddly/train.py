import sys

from typing import Optional

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from envs.griddly.sample_factory_env_wrapper import GriddlyEnvWrapper
from agent import agent

from envs.griddly.power_grid import power_grid_env, power_grid_level_generator

def make_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    lg = power_grid_level_generator.PowerGridLevelGenerator(cfg)
    env = power_grid_env.PowerGridEnvWrapper.make_env(cfg, level_generator=lg)
    return GriddlyEnvWrapper(
        env,
        render_mode=render_mode,
        make_level=lg.make_level_string,
        env_id=env_config.env_id if env_config else 0,
        save_replay_prob=cfg.env_save_replay_prob,
    )

def register_custom_components():
    agent.register_custom_components()
    register_env(power_grid_env.GYM_ENV_NAME, make_env_func)

def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)

    parser.add_argument("--env_num_agents", default=8, type=int,
                        help="number of agents in the environment")
    parser.add_argument("--env_max_steps", default=1000, type=int)
    parser.add_argument("--env_save_replay_prob", default=0.0, type=float)

    power_grid_level_generator.add_env_args(parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status =  run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
