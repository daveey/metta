import json
import hydra

from omegaconf import OmegaConf
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.enjoy import enjoy
from rl_framework.rl_framework import RLFramework
from rl_framework.sample_factory.sample_factory_env_wrapper import SampleFactoryEnvWrapper


def make_env_func(full_env_name, sf_cfg, sf_env_config, render_mode):
    env_cfg = OmegaConf.create(json.loads(sf_cfg.env_cfg))
    env = hydra.utils.instantiate(env_cfg, render_mode=render_mode)
    env = SampleFactoryEnvWrapper(env, env_id=0)
    return env

class SampleFactoryFramework(RLFramework):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sf_args = [
            f"--{k}={v}" for k, v in cfg.sample_factory.items()
        ] + [
            f"--rnn_{k}={v}" for k, v in cfg.agent.rnn.items()
        ]
        register_env(cfg.env.name, make_env_func)
        self.sf_args.append(f"--env={cfg.env.name}")
        self.sf_args.append(
            "--env_cfg=" +
            json.dumps(OmegaConf.to_container(cfg.env, resolve=True)))
        env = hydra.utils.instantiate(cfg.env)
        env = SampleFactoryEnvWrapper(env, env_id=0)
        OmegaConf.set_struct(cfg, False)
        cfg.agent["feature_schema"] = env.gym_env.feature_schema()
        OmegaConf.set_struct(cfg, True)
        self.agent = hydra.utils.instantiate(cfg.agent)

    def setup(self, evaluation=False):
        print("SampleFactory Args: ", self.sf_args)
        sf_parser, cfg = parse_sf_args(self.sf_args, evaluation=evaluation)
        sf_parser.add_argument("--env_cfg", type=str, default=None)
        sf_cfg = parse_full_cfg(sf_parser, self.sf_args)
        return sf_cfg

    def train(self):
        sf_cfg = self.setup()
        status =  run_rl(sf_cfg)
        return status

    def evaluate(self):
        sf_cfg = self.setup(evaluation=True)
        status = enjoy(sf_cfg)
        return status[0]
