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
       'generator:ready', 'generator:resource'
    ]

class FeatureAttnAgentEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        self._grid_shape = obs_space["griddly_obs"].shape[1:3]
        self._griddly_max_features = cfg.agent_max_features
        self._feature_names = FEATURE_NAMES

        self._position_encodings = self._create_position_encodings()
        self._feature_encodings = self._create_feature_encodings(obs_space)

        self._transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.agent_max_features,
                nhead=cfg.agent_num_attention_heads,
                dim_feedforward=cfg.agent_fc_size,
                dropout=cfg.agent_attention_dropout,
                batch_first=True
            ),
            num_layers=cfg.agent_num_attn_layers
        )

        self.register_buffer("objects_mean", torch.zeros(len(self._feature_names), dtype=torch.float64))
        self.register_buffer("objects_var", torch.ones(len(self._feature_names), dtype=torch.float64))
        self.register_buffer("objects_count", torch.ones(1, dtype=torch.float64))

        self.norm_eps: Final[float] = 1e-5
        self.norm_clip: Final[float] = 5.0

        self.encoder_head = nn.Sequential(
            layer_init(nn.Linear(cfg.agent_max_features, cfg.agent_fc_size)),
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
        position_encodings = position_encodings.unsqueeze(0).expand(len(self._feature_names), -1, -1, -1)
        return position_encodings

    def _create_feature_encodings(self, obs_space):
        feature_encodings = []

        def add_feature_encoding(feature_name):
            feature_hash = hash(feature_name)
            feature_encoding = (feature_hash % 20001 - 10000) / 10000  # Map to range [-1, 1]
            feature_encodings.append(feature_encoding)

        for feature_name in self._feature_names:
            for _ in range(np.prod(self._grid_shape)):
                add_feature_encoding(feature_name)

        for key, value in obs_space.items():
            if key != "griddly_obs":
                for i in range(value.shape[0]):
                    feature_name = f"{key}_{i}"
                    add_feature_encoding(feature_name)

        return torch.tensor(feature_encodings, dtype=torch.float32).unsqueeze(1)

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

    def _normalize_objects(self, objects):
        # Create a mask for non-zero values
        non_zero_mask = objects != 0

        if self.training:
            # Compute mean and var only for non-zero values
            batch_mean = objects[non_zero_mask].mean(dim=0)
            batch_var = objects[non_zero_mask].var(dim=0)
            batch_count = non_zero_mask.sum(dim=(0, 1))

            # Update the running mean, var, and count
            self.objects_mean, self.objects_var, self.objects_count = self._update_mean_var_count(
                self.objects_mean, self.objects_var, self.objects_count, batch_mean, batch_var, batch_count
            )

        # Compute the current mean and standard deviation
        current_mean = self.objects_mean
        current_std = torch.sqrt(self.objects_var + self.norm_eps)

        # Normalize all values using the current mean and standard deviation
        normalized_objects = (objects - current_mean.view(1, -1)) / current_std.view(1, -1)

        # Clip the normalized values
        normalized_objects = normalized_objects.clamp(-self.norm_clip, self.norm_clip)

        # Set the normalized values for zero entries back to zero
        normalized_objects[~non_zero_mask] = 0

        return normalized_objects

    def forward(self, obs_dict):
        griddly_obs = obs_dict["griddly_obs"]
        batch_size = griddly_obs.size(0)
        device = griddly_obs.device

        # Add positions to the griddly features
        griddly_features = torch.cat([
            griddly_obs.unsqueeze(-1),
            self._position_encodings.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(device)
        ], dim=-1)
        griddly_features = griddly_features.permute(0, 1, 4, 2, 3).reshape(batch_size, -1, griddly_features.size(2) * griddly_features.size(3))

        # Gather non-griddly feature values
        non_griddly_values = torch.cat([value.view(batch_size, -1) for key, value in obs_dict.items() if key != "griddly_obs"], dim=1)
        non_griddly_features = torch.cat([torch.zeros(batch_size, non_griddly_values.size(1), 2, device=device), non_griddly_values.unsqueeze(2)], dim=2)

        # Reshape non-griddly features to match the size of griddly features in dimension 1
        non_griddly_features = non_griddly_features.view(batch_size, -1, 3)
        non_griddly_features = non_griddly_features.repeat(1, griddly_features.size(1) // non_griddly_features.size(1) + 1, 1)[:, :griddly_features.size(1), :]

        # Concatenate griddly and non-griddly features
        all_features = torch.cat([griddly_features, non_griddly_features], dim=-1)

        # Reshape feature encodings to match the size of all_features in dimension 2
        feature_encodings = self._feature_encodings.unsqueeze(0).expand(batch_size, -1, -1).repeat(1, 1, all_features.size(1) // self._feature_encodings.size(0) + 1)[:, :, :all_features.size(1)].to(device)

        # Add feature encodings
        all_features = torch.cat([all_features, feature_encodings], dim=-1)

        # Filter out non-zero values and normalize
        non_zero_mask = all_features[:, :, -2] != 0
        all_features = all_features[non_zero_mask]
        all_features[:, -2] = self._normalize_objects(all_features[:, -2])

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
        parser.add_argument("--agent_max_features", default=32, type=int, help="Max number of griddly features")
        parser.add_argument("--agent_num_fc_layers", default=4, type=int, help="Number of encoder fc layers")
        parser.add_argument("--agent_fc_size", default=512, type=int, help="Size of the FC layer")
        parser.add_argument("--agent_num_attention_heads", default=8, type=int, help="Number of attention heads")
        parser.add_argument("--agent_num_attn_layers", default=4, type=int, help="Number of transformer layers")
        parser.add_argument("--agent_attention_dropout", default=0.05, type=float, help="Attention dropout")
