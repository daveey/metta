import argparse
from ast import arg
import sys

from typing import Optional

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.enjoy import enjoy
from envs.griddly.sample_factory_env_wrapper import GriddlyEnvWrapper
from agent.agent_factory import AgentFactory
from envs.griddly.power_grid import power_grid_env, power_grid_level_generator
import configs.configs

def make_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    lg = power_grid_level_generator.PowerGridLevelGenerator(cfg)
    env = power_grid_env.PowerGridEnv(lg, render_mode=render_mode)
    return GriddlyEnvWrapper(
        env,
        render_mode=render_mode,
        make_level=lg.make_level_string,
        env_id=env_config.env_id if env_config else 0,
    )

def dedupe_args(args):
    deduped_args = {}
    flags = []
    for arg in args:
        if "=" in arg:
            key, value = arg.split('=')
            deduped_args[key] = value
        else:
            flags.append(arg)
    return [f"{key}={value}" for key, value in deduped_args.items()] + flags

def parse_args(argv=None, evaluation=False):
    # Process the --configs argument first
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[], type=str)
    args, sf_args = parser.parse_known_args(argv)

    configs_args = []
    for s in args.config:
        configs_args += configs.configs.__dict__[s]
    sf_args = configs_args + sf_args

    # Then process the --agent argument, which could come from --configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="object_embedding_agent", type=str)
    args, sf_args = parser.parse_known_args(sf_args)
    agent_factory = AgentFactory()
    agent = agent_factory.create_agent(args.agent)

    # Register the environment
    register_env(power_grid_env.GYM_ENV_NAME, make_env_func)
    sf_args.append(f"--env={power_grid_env.GYM_ENV_NAME}")

    # Add the env and agent args
    sf_parser, cfg = parse_sf_args(sf_args, evaluation=evaluation)
    power_grid_level_generator.add_env_args(sf_parser)
    agent.add_args(sf_parser)

    sf_args = dedupe_args(sf_args)
    cfg = parse_full_cfg(sf_parser, sf_args)
    return cfg

def train(argv = None):
    cfg = parse_args(argv)
    status =  run_rl(cfg)
    return status

def evaluate(argv = None):
    cfg = parse_args(argv, evaluation=True)
    status = enjoy(cfg)
    return status[0]
