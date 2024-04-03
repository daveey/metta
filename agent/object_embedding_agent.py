from __future__ import annotations
from venv import logger
from cv2 import log


import numpy as np
from torch import nn
import torch

from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
from agent.grid_encoder import GridEncoder
from agent.util import layer_init

from agent.sample_factory_agent import SampleFactoryAgent


class ObjectEmeddingAgentEncoder(GridEncoder):

    def __init__(self, cfg, obs_space):
        super().__init__(cfg, obs_space)

        # Precompute position encodings and padding
        self._features_padding = torch.zeros(
            1, # batch
            self._griddly_max_features - self._num_grid_features - 2, # +2 for position encodings
            *self._grid_shape)

        position_encodings = self._create_position_encodings()
        self._position_and_padding = torch.cat([
            position_encodings,
            self._features_padding], dim=1)

        self._num_object_features = self._griddly_max_features

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

    def forward(self, obs_dict):
        grid_obs = self._grid_obs(obs_dict)
        batch_size = grid_obs.size(0)

        # Pad features to fixed size
        pos_and_padding = self._position_and_padding.expand(batch_size, -1, -1, -1)
        pos_and_padding = pos_and_padding.to(grid_obs.device)

        grid_obs = torch.cat([pos_and_padding, grid_obs], dim=1)

        # create one big batch of objects (batch_size * grid_size, num_features)
        object_obs = grid_obs.permute(0, 2, 3, 1).reshape(-1, self._num_object_features)

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


class ObjectEmeddingAgentDecoder(MlpDecoder):
    pass

class ObjectEmeddingAgent(SampleFactoryAgent):
    def encoder_cls(self):
        return ObjectEmeddingAgentEncoder

    def decoder_cls(self):
        return ObjectEmeddingAgentDecoder

    def add_args(self, parser):
        parser.add_argument("--agent_max_features", default=50, type=int, help="Max number of griddly features")
        parser.add_argument("--agent_fc_layers", default=4, type=int, help="Number of encoder fc layers")
        parser.add_argument("--agent_fc_size", default=512, type=int, help="Size of the FC layer")
        parser.add_argument("--agent_embedding_size", default=64, type=int, help="Size of each feature embedding")
        parser.add_argument("--agent_embedding_layers", default=3, type=int, help="Size of each feature embedding")
