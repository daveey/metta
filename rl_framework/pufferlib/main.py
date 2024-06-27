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


def load_config(parser, config_path='rl_framework/pufferlib/config.yaml'):
    '''Just a fancy config loader. Populates argparse from
    yaml + env/policy fn signatures to give you a nice
    --help menu + some limited validation of the config'''
    args, _ = parser.parse_known_args()
    env_name, pkg_name = args.env, args.pkg

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if 'default' not in config:
        raise ValueError('Deleted default config section?')
    if env_name not in config:
        raise ValueError(f'{env_name} not in config\n'
            'It might be available through a parent package, e.g.\n'
            '--config atari --env MontezumasRevengeNoFrameskip-v4.')

    default = config['default']
    env_config = config[env_name or pkg_name]
    pkg_name = pkg_name or env_config.get('package', env_name)
    pkg_config = config[pkg_name]
    # TODO: Check if actually installed
    make_name = env_config.get('env_name', None)
    make_env_args = [make_name] if make_name else []
    make_env = env_module.env_creator(*make_env_args)
    make_env_args = pufferlib.utils.get_init_args(make_env)
    policy_args = {} #pufferlib.utils.get_init_args(env_module.Policy)
    rnn_args = {} #pufferlib.utils.get_init_args(env_module.Recurrent)
    fn_sig = dict(env=make_env_args, policy=policy_args, rnn=rnn_args)
    config = vars(parser.parse_known_args()[0])

    valid_keys = 'env policy rnn train sweep'.split()
    for key in valid_keys:
        fn_subconfig = fn_sig.get(key, {})
        env_subconfig = env_config.get(key, {})
        pkg_subconfig = pkg_config.get(key, {})
        # Priority env->pkg->default->fn config
        config[key] = {**fn_subconfig, **default[key],
            **pkg_subconfig, **env_subconfig}

    for name in valid_keys:
        sub_config = config[name]
        for key, value in sub_config.items():
            data_key = f'{name}.{key}'
            cli_key = f'--{data_key}'.replace('_', '-')
            if isinstance(value, bool) and value is False:
                parser.add_argument(cli_key, default=value, action='store_true')
            elif isinstance(value, bool) and value is True:
                data_key = f'{name}.no_{key}'
                cli_key = f'--{data_key}'.replace('_', '-')
                parser.add_argument(cli_key, default=value, action='store_false')
            else:
                parser.add_argument(cli_key, default=value, type=type(value))

            config[name][key] = getattr(parser.parse_known_args()[0], data_key)
        config[name] = pufferlib.namespace(**config[name])

    pufferlib.utils.validate_args(make_env.func if isinstance(make_env, functools.partial) else make_env, config['env'])
    #pufferlib.utils.validate_args(env_module.Policy, config['policy'])

    if 'use_rnn' in env_config:
        config['use_rnn'] = env_config['use_rnn']
    elif 'use_rnn' in pkg_config:
        config['use_rnn'] = pkg_config['use_rnn']
    else:
        config['use_rnn'] = default['use_rnn']

    parser.add_argument('--use_rnn', default=False, action='store_true',
        help='Wrap policy with an RNN')
    config['use_rnn'] = config['use_rnn'] or parser.parse_known_args()[0].use_rnn
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.parse_args()
    wandb_name = make_name or env_name
    config['env_name'] = env_name
    config['exp_id'] = args.exp_id or args.env + '-' + str(uuid.uuid4())[:8]
    return wandb_name, pkg_name, pufferlib.namespace(**config), env_module, make_env, make_policy



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


@hydra.main(version_base=None, config_path="../../configs", config_name="configs")
def main(cfg):

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
        cfg = {k: v for k, v in cfg.env.items() if k != 'name'}
        creator = lambda: make_env(cfg)
        pufferlib.vector.autotune(creator, batch_size=args.train.env_batch_size)
    elif args.mode == 'profile':
        import cProfile
        cProfile.run('train(args, env_module, make_env)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)

if __name__ == '__main__':
    main()