from pdb import set_trace as T

import gym
from omegaconf import OmegaConf
import shimmy
import functools

from env.wrapper.petting_zoo import PettingZooEnvWrapper
import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess
import hydra


def env_creator(name='GDY-MettaGrid'):
    return functools.partial(make, name)

def make(name, cfg: OmegaConf, render_mode='rgb_array'):
    env = hydra.utils.instantiate(cfg, render_mode=render_mode)
    # env = pufferlib.postprocess.EpisodeStats(env)
    env = PettingZooEnvWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env)
