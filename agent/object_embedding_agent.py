from __future__ import annotations
import json
from venv import logger
from chex import dataclass
from cv2 import log


import numpy as np
from omegaconf import OmegaConf
from torch import nn
import torch

from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
from agent.grid_encoder import GridEncoder
from .util import layer_init, make_nn_stack
from agent.sample_factory_agent import SampleFactoryAgent

@dataclass
class ObjectEmeddingAgentEncoderCfg:
    max_features: int = 50
    fc_layers: int = 4
    fc_size: int = 512
    fc_norm: bool = False
    fc_skip: bool = False
    embedding_size: int = 64
    embedding_layers: int = 3
    embedding_norm: bool = False
    embedding_skip: bool = False

class ObjectEmeddingAgentEncoder(GridEncoder):

    def __init__(self, cfg, obs_space):
        super().__init__(cfg, obs_space)

        # Object embedding network
        self.object_embedding = make_nn_stack(
            input_size=self._num_grid_features,
            output_size=self._cfg.embedding_size,
            hidden_sizes=[self._cfg.embedding_size] * self._cfg.embedding_layers,
            nonlinearity=nonlinearity(cfg),
            layer_norm=self._cfg.embedding_norm,
            use_skip=self._cfg.embedding_skip,
        )

        # Additional features size calculation
        all_embeddings_size = (
            self._cfg.embedding_size * np.prod(self._grid_shape) +
            obs_space["global_vars"].shape[0] +
            obs_space["last_action"].shape[0] +
            obs_space["last_reward"].shape[0]
        )

        # Encoder head
        self.encoder_head = make_nn_stack(
            input_size=all_embeddings_size,
            output_size=self._cfg.fc_size,
            hidden_sizes=[self._cfg.fc_size] * self._cfg.fc_layers,
            nonlinearity=nonlinearity(cfg),
            layer_norm=self._cfg.fc_norm,
            use_skip=self._cfg.fc_skip,
        )

        self.encoder_head_out_size = self._cfg.fc_size

    def forward(self, obs_dict):
        grid_obs = self._grid_obs(obs_dict)
        batch_size = grid_obs.size(0)

        # create one big batch of objects (batch_size * grid_size, num_features)
        object_obs = grid_obs.permute(0, 2, 3, 1).reshape(-1, self._num_grid_features)

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

    @classmethod
    def add_args(cls, parser):
        GridEncoder.add_args(parser)

class ObjectEmeddingAgentDecoder(MlpDecoder):
    pass

class ObjectEmeddingAgent(SampleFactoryAgent):
    def encoder_cls(self):
        return ObjectEmeddingAgentEncoder

    def decoder_cls(self):
        return ObjectEmeddingAgentDecoder

