from __future__ import annotations


import numpy as np
from torch import nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.utils.typing import Config


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

class GriddlyDecoder(Decoder):
    def __init__(self, cfg: Config, decoder_input_size: int):
        super().__init__(cfg)
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(decoder_input_size, 256), std=0.01),
            nn.ReLU()
        )

    def get_out_size(self) -> int:
        return 256

    def forward(self, core_output):
        return self.actor_head(core_output)

