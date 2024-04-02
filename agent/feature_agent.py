from __future__ import annotations

import numpy as np
from torch import nn
import torch
from torch import Tensor
from typing import Final

from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
from agent.sample_factory_agent import SampleFactoryAgent
from agent.util import layer_init


class FeatureAgentEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        self._feature_embedding = nn.ModuleDict({
            k: nn.Sequential(
                layer_init(nn.Linear(np.prod(v.shape), cfg.agent_feature_embedding_size)),
                nonlinearity(cfg))
            for k, v in obs_space.items()
        })

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(cfg.agent_feature_embedding_size, cfg.agent_fc_size)),
            nonlinearity(cfg),
            *[nn.Sequential(
                layer_init(nn.Linear(cfg.agent_fc_size, cfg.agent_fc_size)),
                nonlinearity(cfg)
              ) for _ in range(cfg.agent_num_fc_layers)]
        )
        self.encoder_out_size = cfg.agent_fc_size

    def forward(self, obs_dict):
        batch_size = obs_dict["agent"].size(0)
        device = obs_dict["agent"].device
        embeddings = [
            self._feature_embedding[k](v.view(v.size(0), -1))
            for k, v in obs_dict.items()
        ]

        all_features = torch.stack(embeddings, dim=1).to(device)
        max_features = torch.max(all_features, dim=1).values
        x = self.encoder(max_features)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

class FeatureAgentDecoder(MlpDecoder):
    pass

class FeatureAgent(SampleFactoryAgent):
    def encoder_cls(self):
        return FeatureAgentEncoder

    def decoder_cls(self):
        return FeatureAgentDecoder

    def add_args(self, parser):
        parser.add_argument("--agent_feature_embedding_size", default=32, type=int, help="Max number of griddly features")
        parser.add_argument("--agent_num_fc_layers", default=4, type=int, help="Number of encoder fc layers")
        parser.add_argument("--agent_fc_size", default=512, type=int, help="Size of the FC layer")
        parser.add_argument("--agent_num_attention_heads", default=1, type=int, help="Number of attention heads")
        parser.add_argument("--agent_num_attn_layers", default=4, type=int, help="Number of transformer layers")
        parser.add_argument("--agent_attention_dropout", default=0.05, type=float, help="Attention dropout")
