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


FEATURE_NAMES = [
    'agent', 'altar', 'charger', 'generator', 'wall',
     'agent:dir', 'agent:energy', 'agent:frozen',
     'agent:id', 'agent:inv:1', 'agent:inv:2', 'agent:inv:3',
    'agent:shield', 'agent:species', 'altar:ready',
       'charger:bonus', 'charger:input:1', 'charger:input:2',
       'charger:input:3', 'charger:output', 'charger:ready',
       'generator:ready', 'generator:resource', 'kinship'
    ]

class FeatureAttnAgentEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        self._grid_shape = obs_space["griddly_obs"].shape[1:3]

        self._other_feature_names = []
        for key, value in obs_space.items():
            if key != "griddly_obs" and key != "kinship":
                for i in range(np.prod(value.shape)):
                    fn = f"{key}_{i}"
                    self._other_feature_names.append(fn)
        self._griddly_feature_names = FEATURE_NAMES
        self._feature_names = self._other_feature_names + self._griddly_feature_names

        nes = self._create_feature_encodings(obs_space)
        pes = self._create_position_encodings()
        position_padding = torch.zeros((1, nes.shape[1] - pes.shape[1], 2))
        padded_pes = torch.cat([position_padding, pes], dim=1)
        self._feature_encodings = torch.cat([nes, padded_pes], dim=2)

        self._feature_embedding = nn.Linear(4 , cfg.agent_feature_embedding_size)

        self._transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.agent_feature_embedding_size,
                nhead=cfg.agent_num_attention_heads,
                dim_feedforward=cfg.agent_fc_size,
                dropout=cfg.agent_attention_dropout,
                batch_first=True
            ),
            num_layers=cfg.agent_num_attn_layers
        )

        self.register_buffer("features_mean", torch.zeros(nes.shape[1], dtype=torch.float64))
        self.register_buffer("features_var", torch.ones(nes.shape[1], dtype=torch.float64))
        self.register_buffer("features_count", torch.ones(nes.shape[1], dtype=torch.float64))

        self.norm_eps: Final[float] = 1e-5
        self.norm_clip: Final[float] = 5.0

        self.encoder_head = nn.Sequential(
            layer_init(nn.Linear(cfg.agent_feature_embedding_size, cfg.agent_fc_size)),
            nonlinearity(cfg),
            *[nn.Sequential(
                layer_init(nn.Linear(cfg.agent_fc_size, cfg.agent_fc_size)),
                nonlinearity(cfg)
              ) for _ in range(cfg.agent_num_fc_layers)]
        )
        self.encoder_head_out_size = cfg.agent_fc_size

    def _create_position_encodings(self):
        x = torch.linspace(-1, 1, self._grid_shape[0])
        y = torch.linspace(-1, 1, self._grid_shape[1])
        pos_x, pos_y = torch.meshgrid(x, y, indexing='xy')
        position_encodings = torch.stack((pos_x, pos_y), dim=-1)
        position_encodings = position_encodings.unsqueeze(0).expand(len(self._griddly_feature_names), -1, -1, -1)
        return position_encodings.reshape(1, -1, 2)

    def _create_feature_encodings(self, obs_space):
        fe = lambda x: (hash(x) % 20001 - 10000) / 10000
        griddly_fes = [ fe(f) for f in self._griddly_feature_names ]
        griddly_fes *= np.prod(self._grid_shape)

        other_fes = [ fe(n) for n in self._other_feature_names]
        self._feature_names += self._griddly_feature_names
        return torch.tensor(other_fes + griddly_fes, dtype=torch.float32).view(1, -1, 1)

    @staticmethod
    @torch.jit.script
    def _update_mean_var_count(
        mean: Tensor, var: Tensor, count: Tensor, batch_mean: Tensor, batch_var: Tensor, batch_count: int
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count

    def _normalize_features(self, features):
        if self.training:
            # Compute mean and var only for non-zero values
            batch_mean = features.mean(dim=-1)
            batch_var = features.var(dim=-1)
            batch_count = features.sum(dim=(0, 1))

            # Update the running mean, var, and count
            self.objects_mean, self.objects_var, self.objects_count = self._update_mean_var_count(
                self.objects_mean, self.objects_var, self.objects_count, batch_mean, batch_var, batch_count
            )

        # Compute the current mean and standard deviation
        current_mean = self.objects_mean
        current_std = torch.sqrt(self.objects_var + self.norm_eps)

        # Normalize all values using the current mean and standard deviation
        normalized_objects = (features - current_mean.view(1, -1)) / current_std.view(1, -1)

        # Clip the normalized values
        normalized_objects = normalized_objects.clamp(-self.norm_clip, self.norm_clip)

        return normalized_objects

    def forward(self, obs_dict):
        griddly_obs = torch.cat([obs_dict["griddly_obs"], obs_dict["kinship"]], dim=1)
        batch_size = griddly_obs.size(0)
        device = griddly_obs.device

        griddly_features = griddly_obs.view(batch_size, -1, 1)

        # Gather non-griddly feature values
        non_griddly_values = torch.cat([value.view(batch_size, -1) for key, value in obs_dict.items() if key != "griddly_obs" and key != "kinship"], dim=1)
        non_griddly_features = non_griddly_values.view(batch_size, -1, 1)

        # Concatenate griddly and non-griddly features
        all_features = torch.cat([non_griddly_features, griddly_features], dim=1)

        # Add feature encodings
        fes = self._feature_encodings.expand(batch_size, -1, 3).to(device)
        all_features = torch.cat([fes, all_features], dim=2)
        all_features = self._feature_embedding(
            all_features.view(-1, 4)).view(batch_size, all_features.shape[1], -1)

        # Filter out non-zero values and normalize
        # all_features = self._normalize_features(all_features)

        # Apply transformer layers
        all_features = self._transformer_layers(all_features)

        x = self.encoder_head(all_features.mean(dim=1))
        return x.view(-1, self.encoder_head_out_size)

    def get_out_size(self) -> int:
        return self.encoder_head_out_size

class FeatureAttnAgentDecoder(MlpDecoder):
    pass

class FeatureAttnAgent(SampleFactoryAgent):
    def encoder_cls(self):
        return FeatureAttnAgentEncoder

    def decoder_cls(self):
        return FeatureAttnAgentDecoder

    def add_args(self, parser):
        parser.add_argument("--agent_feature_embedding_size", default=32, type=int, help="Max number of griddly features")
        parser.add_argument("--agent_num_fc_layers", default=4, type=int, help="Number of encoder fc layers")
        parser.add_argument("--agent_fc_size", default=512, type=int, help="Size of the FC layer")
        parser.add_argument("--agent_num_attention_heads", default=1, type=int, help="Number of attention heads")
        parser.add_argument("--agent_num_attn_layers", default=4, type=int, help="Number of transformer layers")
        parser.add_argument("--agent_attention_dropout", default=0.05, type=float, help="Attention dropout")
