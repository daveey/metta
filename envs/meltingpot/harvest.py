
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from meltingpot import substrate

from envs.meltingpot.melting_pot_env import MeltingPotEnv
from rl_framework.sample_factory.sample_factory_env_wrapper import SampleFactoryEnvWrapper

class HarvestEnv(SampleFactoryEnvWrapper):
    def __init__(self, env_id:int=0, **cfg):
        self._cfg = cfg
        env_name = "allelopathic_harvest__open"
        env_config = substrate.get_config(env_name)
        env = MeltingPotEnv(env_config, max_cycles=1000)
        super().__init__(env, env_id=env_id)
