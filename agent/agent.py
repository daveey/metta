from __future__ import annotations
from os import error
from gym import make
import gym

from sample_factory.algo.utils.tensor_dict import TensorDict

import numpy as np
from torch import nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.core import ModelCore, ModelCoreRNN
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.model.model_utils import nonlinearity
from sample_factory.utils.typing import Config
import torch
from sample_factory.algo.utils.context import global_model_factory
from agent.predicting_actor_critic import make_actor_critic_func

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Simple function to init layers
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class GriddlyEncoder(Encoder):
    """Just an example of how to use a custom model component."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        obs_shape = obs_space["obs"].shape

        self._num_objects = obs_shape[0]

        linear_flatten = np.prod(obs_shape[1:]) * 64

        self.conv_head = nn.Sequential(
            layer_init(nn.Conv2d(self._num_objects, 32, 3, padding=1)),
            nonlinearity(cfg),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nonlinearity(cfg),
            nn.Flatten(),
            layer_init(nn.Linear(linear_flatten, 1024)),
            nonlinearity(cfg),
            layer_init(nn.Linear(1024, 512)),
            nonlinearity(cfg),
        )
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict["obs"]
        x = self.conv_head(main_obs)
        x = x.view(-1, self.conv_head_out_size)
        return x

    def get_out_size(self) -> int:
        return self.conv_head_out_size

class GriddlyDecoder(MlpDecoder):
    pass


def register_custom_components():
    global_model_factory().register_encoder_factory(GriddlyEncoder)
    global_model_factory().register_decoder_factory(GriddlyDecoder)
    global_model_factory().register_actor_critic_factory(make_actor_critic_func)

