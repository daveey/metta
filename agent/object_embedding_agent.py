from __future__ import annotations
from venv import logger
from cv2 import log


import numpy as np
from torch import nn
import torch

from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
from agent.grid_encoder import GridEncoder
from .util import layer_init, make_nn_stack
from agent.sample_factory_agent import SampleFactoryAgent

class ObjectEmeddingAgentEncoder(GridEncoder):

    def __init__(self, cfg, obs_space):
        super().__init__(cfg, obs_space)

        # Object embedding network
        self.object_embedding = make_nn_stack(
            input_size=self._num_grid_features,
            output_size=cfg.agent_embedding_size,
            hidden_sizes=[cfg.agent_embedding_size] * cfg.agent_embedding_layers,
            nonlinearity=nonlinearity(cfg),
            layer_norm=cfg.agent_embedding_norm,
            use_skip=cfg.agent_embedding_skip,
            fixup=cfg.agent_embedding_fixup
        )

        # Additional features size calculation
        all_embeddings_size = (
            cfg.agent_embedding_size * np.prod(self._grid_shape) +
            obs_space["global_vars"].shape[0] +
            obs_space["last_action"].shape[0] +
            obs_space["last_reward"].shape[0]
        )

        # Encoder head
        self.encoder_head = make_nn_stack(
            input_size=all_embeddings_size,
            output_size=cfg.agent_fc_size,
            hidden_sizes=[cfg.agent_fc_size] * cfg.agent_fc_layers,
            nonlinearity=nonlinearity(cfg),
            layer_norm=cfg.agent_fc_norm,
            use_skip=cfg.agent_fc_skip,
            fixup=cfg.agent_fc_fixup
        )

        self.encoder_head_out_size = cfg.agent_fc_size

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

    def add_args(self, parser):
        ObjectEmeddingAgentEncoder.add_args(parser)
        parser.add_argument("--agent_max_features", default=50, type=int, help="Max number of griddly features")

        parser.add_argument("--agent_fc_layers", default=4, type=int, help="Number of encoder fc layers")
        parser.add_argument("--agent_fc_size", default=512, type=int, help="Size of the FC layer")
        parser.add_argument("--agent_fc_norm", default=False, type=bool, help="Use layer norms")
        parser.add_argument("--agent_fc_skip", default=False, type=bool, help="Use skip connections")
        parser.add_argument("--agent_fc_fixup", default=False, type=bool, help="Use fixup scaling")


        parser.add_argument("--agent_embedding_size", default=64, type=int, help="Size of each feature embedding")
        parser.add_argument("--agent_embedding_layers", default=3, type=int, help="Size of each feature embedding")
        parser.add_argument("--agent_embedding_norm", default=False, type=bool, help="Use layer norms")
        parser.add_argument("--agent_embedding_skip", default=False, type=bool, help="Use skip connections")
        parser.add_argument("--agent_embedding_fixup", default=False, type=bool, help="Use fixup scaling")

