from __future__ import annotations


import numpy as np
from torch import nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
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
        layers = []

        for i in range(cfg.agent_conv_layers):
            layers.append(layer_init(
                nn.Conv2d(self._num_objects if i == 0 else
                          cfg.agent_conv_size, cfg.agent_conv_size,
                          3, padding=1)))
            layers.append(nonlinearity(cfg))

        if cfg.agent_conv_layers > 0:
            linear_flatten = np.prod(obs_shape[1:]) * cfg.agent_conv_size
        else:
            linear_flatten = np.prod(obs_shape)

        layers.append(nn.Flatten())
        for i in range(cfg.agent_fc_layers):
            layers.append(layer_init(nn.Linear(linear_flatten if i == 0 else
                                               cfg.agent_fc_size, cfg.agent_fc_size)))
            layers.append(nonlinearity(cfg))

        self.encoder_head = nn.Sequential(*layers)
        self.encoder_head_out_size = calc_num_elements(self.encoder_head, obs_shape)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict["obs"]
        x = self.encoder_head(main_obs)
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
    parser.add_argument("--agent_conv_layers", default=2, type=int, help="Number of encoder conv layers")
    parser.add_argument("--agent_conv_size", default=64, type=int, help="Size of the FC layer")
    parser.add_argument("--agent_fc_layers", default=2, type=int, help="Number of encoder fc layers")
    parser.add_argument("--agent_fc_size", default=512, type=int, help="Size of the FC layer")
