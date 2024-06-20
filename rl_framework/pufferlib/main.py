from calendar import c
from pdb import set_trace as T
import functools
import argparse
import shutil
import yaml
import uuid
import sys
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl
import hydra
from omegaconf import OmegaConf

import rl_framework.pufferlib as env_module

from rich_argparse import RichHelpFormatter
from rich.traceback import install
from rich.console import Console

from . import clean_pufferl


def make_policy(env, env_module, args):
    policy_args = pufferlib.utils.get_init_args(env_module.Policy)
    policy = env_module.Policy(env, **policy_args)
    if args.use_rnn:
        policy = env_module.Recurrent(env, policy, **args.rnn)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args.train.device)

def init_wandb(args, name, id=None, resume=True):
    #os.environ["WANDB_SILENT"] = "true"
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb.project,
        entity=args.wandb.entity,
        group=args.wandb.group,
        config={
            'cleanrl': dict(args.train),
            'env': dict(args.env),
            'policy': dict(args.policy),
            #'recurrent': args.recurrent,
        },
        name=name,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )
    return wandb

def sweep(args, wandb_name, env_module, make_env):
    import wandb
    sweep_id = wandb.sweep(
        sweep=dict(args.sweep),
        project="pufferlib",
    )

    def main():
        try:
            args.exp_name = init_wandb(args, wandb_name, id=args.exp_id)
            # TODO: Add update method to namespace
            print(wandb.config.train)
            args.train.__dict__.update(dict(wandb.config.train))
            args.track = True
            train(args, env_module, make_env)
        except Exception as e:
            import traceback
            traceback.print_exc()

    wandb.agent(sweep_id, main, count=100)

def train(cfg, env_module, make_env):
    args = cfg.pufferlib
    if args.wandb.track:
        wandb = init_wandb(args, "xcxc_wandb", id=args.exp_id)

    vec = args.vectorization
    if vec == 'serial':
        vec = pufferlib.vector.Serial
    elif vec == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif vec == 'ray':
        vec = pufferlib.vector.Ray
    else:
        raise ValueError(f'Invalid --vector (serial/multiprocessing/ray).')

    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=dict(cfg = dict(**cfg.env)),
        num_envs=args.train.num_envs,
        num_workers=args.train.num_workers,
        batch_size=args.train.env_batch_size,
        zero_copy=args.train.zero_copy,
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, env_module, args)
    train_config = args.train
    train_config.track = args.wandb.track
    train_config.device = args.train.device
    train_config.env = cfg.env.name

    if args.backend == 'clean_pufferl':
        data = clean_pufferl.create(train_config, vecenv, policy, wandb=args.wandb)

        while data.global_step < args.train.total_timesteps:
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

    elif args.backend == 'sb3':
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.env_util import make_vec_env
        from sb3_contrib import RecurrentPPO

        envs = make_vec_env(lambda: make_env(**args.env),
            n_envs=args.train.num_envs, seed=args.train.seed, vec_env_cls=DummyVecEnv)

        model = RecurrentPPO("CnnLstmPolicy", envs, verbose=1,
            n_steps=args.train.batch_rows*args.train.bptt_horizon,
            batch_size=args.train.batch_size, n_epochs=args.train.update_epochs,
            gamma=args.train.gamma
        )

        model.learn(total_timesteps=args.train.total_timesteps)

@hydra.main(version_base=None, config_path="../../configs", config_name="configs")
def main(cfg):
    install(show_locals=False) # Rich tracebacks

    make_env = env_module.env_creator(cfg.env.name)
    make_env_args = pufferlib.utils.get_init_args(make_env)
    rnn_args = pufferlib.utils.get_init_args(env_module.Recurrent)

    wandb_name = cfg.env.name
    args = cfg.pufferlib

    if args.baseline:
        assert args.mode in ('train', 'eval', 'evaluate')
        args.wandb.track = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args.exp_id = f'puf-{version}-{cfg.env.name}'
        args.wandb.group = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args.exp_id}', ignore_errors=True)
        run = init_wandb(cfg, args.exp_id, resume=False)
        if args.mode in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{cfg.env.name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args.eval_model_path = os.path.join(data_dir, model_file)

    if args.mode == 'train':
        train(cfg, env_module, make_env)
    elif args.mode in ('eval', 'evaluate'):
        try:
            clean_pufferl.rollout(
                make_env,
                {cfg: cfg.env},
                agent_creator=make_policy,
                agent_kwargs={'env_module': env_module, 'args': args},
                model_path=args.eval_model_path,
                render_mode=args.render_mode,
                device=args.train.device
            )
        except KeyboardInterrupt:
            os._exit(0)
    elif args.mode == 'sweep':
        sweep(cfg, wandb_name, env_module, make_env)
    elif args.mode == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args.train.env_batch_size)
    elif args.mode == 'profile':
        import cProfile
        cProfile.run('train(args, env_module, make_env)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)

if __name__ == '__main__':
    main()
