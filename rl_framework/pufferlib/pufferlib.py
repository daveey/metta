from typing import Dict
from omegaconf import OmegaConf
from env.wrapper.petting_zoo import PettingZooEnvWrapper
from rl_framework.rl_framework import RLFramework
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl
import hydra
from omegaconf import OmegaConf

from rich.console import Console

from . import puffer_agent_wrapper

from . import clean_pufferl

def make_env_func(cfg: OmegaConf, render_mode='rgb_array'):
    env = hydra.utils.instantiate(cfg, render_mode=render_mode)
    # env = pufferlib.postprocess.EpisodeStats(env)
    env = PettingZooEnvWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env)

class PufferLibFramework(RLFramework):
    def __init__(self, cf: Dict, **puffer_args):
        cfg = OmegaConf.create(cf)
        super().__init__(cfg)
        self.puffer_cfg = OmegaConf.create(puffer_args)
        self.wandb = None
        if self.puffer_cfg.wandb.track:
            self.wandb = init_wandb(self.cfg, resume=True)

    def train(self):
        pcfg = self.puffer_cfg
        vec = pcfg.vectorization
        if vec == 'serial':
            vec = pufferlib.vector.Serial
        elif vec == 'multiprocessing':
            vec = pufferlib.vector.Multiprocessing
        elif vec == 'ray':
            vec = pufferlib.vector.Ray
        else:
            raise ValueError(f'Invalid --vector (serial/multiprocessing/ray).')

        vecenv = pufferlib.vector.make(
            make_env_func,
            env_kwargs=dict(cfg = dict(**self.cfg.env)),
            num_envs=pcfg.train.num_envs,
            num_workers=pcfg.train.num_workers,
            batch_size=pcfg.train.env_batch_size,
            zero_copy=pcfg.train.zero_copy,
            backend=vec,
        )
        policy = puffer_agent_wrapper.make_policy(vecenv.driver_env, self.cfg)
        data = clean_pufferl.create(pcfg.train, vecenv, policy, wandb=self.wandb)

        while data.global_step < pcfg.train.total_timesteps:
            try:
                clean_pufferl.evaluate(data)
                clean_pufferl.train(data)
            except KeyboardInterrupt:
                clean_pufferl.close(data)
                os._exit(0)
            except Exception:
                Console().print_exception()
                os._exit(0)

        clean_pufferl.evaluate(data)
        clean_pufferl.close(data)

    def evaluate(self):
        clean_pufferl.rollout(
            make_env_func,
            env_kwargs=dict(cfg = dict(**self.cfg.env)),
            agent_creator=puffer_agent_wrapper.make_policy,
            agent_kwargs={'cfg': self.cfg},
            model_path=self.puffer_cfg.eval_model_path,
            render_mode=self.puffer_cfg.render_mode,
            device=self.puffer_cfg.train.device
        )

def init_wandb(cfg: OmegaConf, resume=True):
    #os.environ["WANDB_SILENT"] = "true"
    import wandb
    wandb.init(
        id=cfg.experiment or wandb.util.generate_id(),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )
    return wandb