from pdb import set_trace as T

import torch
from torch import nn

import pufferlib
import pufferlib.models
import pufferlib.pytorch

from agent.metta_agent import MettaAgent

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class Policy(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        obs_space = env.env._gym_env.observation_space
        atn_space = env.env._gym_env.action_space

        cfg.observation_encoders.grid_obs.feature_names = env.env._gym_env.grid_features
        cfg.observation_encoders.global_vars.feature_names = env.env._gym_env.global_features
        self.model = MettaAgent(obs_space, atn_space, **cfg)
        self.atn_type = nn.Linear(cfg.core.rnn_size, 6)
        self.atn_param = nn.Linear(cfg.core.rnn_size, 10)

    def forward(self, obs):
        x = self.encode_observations(obs)
        return self.decode_actions(x, None)

    def encode_observations(self, flat_obs):
        x = pufferlib.pytorch.nativize_tensor(flat_obs, self.dtype)
        return self.model.encode_observations(x), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = [self.atn_type(flat_hidden), self.atn_param(flat_hidden)]
        value = self.model._critic_linear(flat_hidden)
        return action, value
