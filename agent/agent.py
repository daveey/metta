from __future__ import annotations
from turtle import pos


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

        self._grid_shape = obs_space["griddly_obs"].shape[1:3]
        self._griddly_max_features = cfg.agent_max_features

        # Precompute position encodings and padding
        self._features_padding = torch.zeros(
            1, # batch
            self._griddly_max_features - obs_space["griddly_obs"].shape[0],
            *self._grid_shape)

        position_encodings = self._create_position_encodings()
        self._position_and_padding = torch.cat([
            position_encodings,
            self._features_padding], dim=1)
        self._cached_pos_and_padding = None

        self._num_object_features = self._griddly_max_features + 2 # position encodings

        # Object embedding network
        self.object_embedding = nn.Sequential(
            layer_init(nn.Linear(self._num_object_features, cfg.agent_embedding_size)),
            nonlinearity(cfg),
            *[
                nn.Sequential(
                layer_init(nn.Linear(cfg.agent_embedding_size, cfg.agent_embedding_size)),
                nonlinearity(cfg)
              ) for _ in range(cfg.agent_embedding_layers)]
        )

        # Additional features size calculation
        all_embeddings_size = (
            cfg.agent_embedding_size * np.prod(self._grid_shape) +
            obs_space["global_vars"].shape[0] +
            obs_space["last_action"].shape[0] +
            obs_space["last_reward"].shape[0]
        )

        # Encoder head
        self.encoder_head = nn.Sequential(
            layer_init(nn.Linear(all_embeddings_size, cfg.agent_fc_size)),
            nonlinearity(cfg),
            *[nn.Sequential(
                layer_init(nn.Linear(cfg.agent_fc_size, cfg.agent_fc_size)),
                nonlinearity(cfg)
              ) for _ in range(cfg.agent_fc_layers)]
        )
        self.encoder_head_out_size = cfg.agent_fc_size

    # generate position encodings, shaped (1, 2, width, height)
    def _create_position_encodings(self):
        x = torch.linspace(-1, 1, self._grid_shape[0])
        y = torch.linspace(-1, 1, self._grid_shape[1])
        pos_x, pos_y = torch.meshgrid(x, y, indexing='xy')
        position_encodings = torch.stack((pos_x, pos_y), dim=-1)
        return position_encodings.unsqueeze(0).permute(0, 3, 1, 2)

    def forward(self, obs_dict):
        griddly_obs = obs_dict["griddly_obs"]
        batch_size = griddly_obs.size(0)

        # Pad features to fixed size
        pos_and_padding = self._position_and_padding.expand(batch_size, -1, -1, -1)
        pos_and_padding = pos_and_padding.to(griddly_obs.device)

        griddly_obs = torch.cat([pos_and_padding, griddly_obs], dim=1)

        # create one big batch of objects (batch_size * grid_size, num_features)
        object_obs = griddly_obs.permute(0, 2, 3, 1).reshape(-1, self._num_object_features)

        # Object embedding
        object_embeddings = self.object_embedding(object_obs).view(batch_size, -1)

        # Additional features
        additional_features = torch.cat([
            obs_dict["global_vars"],
            obs_dict["last_action"].view(batch_size, -1),
            obs_dict["last_reward"].view(batch_size, -1)
        ], dim=1)

        all_obs = torch.cat([object_embeddings, additional_features], dim=1)
        # Final encoding
        x = self.encoder_head(all_obs)
        return x.view(-1, self.encoder_head_out_size)

    def get_out_size(self) -> int:
        return self.encoder_head_out_size

class GriddlyDecoder(MlpDecoder):
    pass

def register_custom_components():
    global_model_factory().register_encoder_factory(GriddlyEncoder)
    global_model_factory().register_decoder_factory(GriddlyDecoder)
    global_model_factory().register_actor_critic_factory(make_actor_critic_func)


def add_args(parser):
    parser.add_argument("--agent_max_features", default=50, type=int, help="Max number of griddly features")
    parser.add_argument("--agent_fc_layers", default=4, type=int, help="Number of encoder fc layers")
    parser.add_argument("--agent_fc_size", default=512, type=int, help="Size of the FC layer")
    parser.add_argument("--agent_embedding_size", default=512, type=int, help="Size of each feature embedding")
    parser.add_argument("--agent_embedding_layers", default=3, type=int, help="Size of each feature embedding")
    parser.add_argument("--agent_attention_size", default=512, type=int, help="Inner size of the attention layer")
    parser.add_argument("--agent_attention_layers", default=3, type=int, help="Number of attention layers")
