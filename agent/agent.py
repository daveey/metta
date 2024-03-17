from __future__ import annotations


import numpy as np
from torch import nn
import torch

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.algo.utils.context import global_model_factory
from agent.predicting_actor_critic import make_actor_critic_func
import torch.nn.functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Simple function to init layers
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class GriddlyEncoder(Encoder):

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self._num_features = (
            np.prod(obs_space["obs"].shape) +
            obs_space["global_vars"].shape[0] +
            obs_space["last_action"].shape[0] +
            obs_space["last_reward"].shape[0] +
            np.prod(obs_space["kinship"].shape)
        )

        self.encoder_head = nn.Sequential(*[
            nn.Flatten(),
            layer_init(nn.Linear(self._num_features, cfg.agent_fc_size)),
            nonlinearity(cfg)
        ] + [
            layer_init(nn.Linear(cfg.agent_fc_size, cfg.agent_fc_size), cfg.agent_fc_size),
            nonlinearity(cfg)
        ] * cfg.agent_fc_layers)

        self.encoder_head_out_size = cfg.agent_fc_size

    def forward(self, obs_dict):
        batch_size = obs_dict["last_action"].size(0)
        features = torch.concat([
                obs_dict["obs"].view(batch_size, -1),
                obs_dict["global_vars"],
                obs_dict["last_action"].view(batch_size, -1),
                obs_dict["last_reward"].view(batch_size, -1),
                obs_dict["kinship"].view(batch_size, -1),
            ], dim=1)
        x = self.encoder_head(features)

        x = x.view(-1, self.encoder_head_out_size)
        return x

    def get_out_size(self) -> int:
        return self.encoder_head_out_size

class GriddlyDecoder(MlpDecoder):
    pass


def register_custom_components():
    global_model_factory().register_encoder_factory(GriddlyEncoder)
    global_model_factory().register_decoder_factory(GriddlyDecoder)
    global_model_factory().register_actor_critic_factory(make_actor_critic_func)


def add_args(parser):
    parser.add_argument("--agent_fc_layers", default=4, type=int, help="Number of encoder fc layers")
    parser.add_argument("--agent_fc_size", default=512, type=int, help="Size of the FC layer")
