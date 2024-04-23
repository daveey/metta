from __future__ import annotations


import numpy as np
from sample_factory.model.encoder import Encoder
from torch import nn
import torch

from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
from agent.grid_encoder import GridEncoder
from .util import make_nn_stack
from agent.sample_factory_agent import SampleFactoryAgent

class FeatureDictEncoder(Encoder):

    def __init__(self, sf_cfg, obs_space, agent_cfg):
        super().__init__(sf_cfg)
        self._cfg = agent_cfg

        assert len(self._cfg.grid_feature_names) == obs_space["grid_obs"].shape[0], \
            f"Number of grid features in config ({len(self._cfg.grid_feature_names)}) " \
            f"does not match the number of grid features in the observation space ({obs_space['grid_obs'].shape[0]})"

        self._grid_shape = obs_space["grid_obs"].shape[1:]
        self._num_grid_features = obs_space["grid_obs"].shape[0]
        embedding_nets = {}
        self._embedding_name_ebd = {}
        for grid_feature_name in self._cfg.grid_feature_names:
            embedding_nets[grid_feature_name] = make_nn_stack(
                input_size=np.prod(self._grid_shape),
                output_size=self._cfg.embedding_size,
                hidden_sizes=[self._cfg.embedding_size] * (self._cfg.embedding_layers -1),
                nonlinearity=nonlinearity(sf_cfg),
            )
            self._embedding_name_ebd[grid_feature_name] = \
                  hash(grid_feature_name) % 100000 / 100000.0
        self._embeddings = nn.ModuleDict(embedding_nets)

        self._embedding_proj = make_nn_stack(
            input_size=self._cfg.embedding_size + 1,
            output_size=self._cfg.embedding_size,
            hidden_sizes=[self._cfg.embedding_size] * (self._cfg.projection_layers - 1),
            nonlinearity=nonlinearity(sf_cfg),
        )

        all_embeddings_size = (
            self._cfg.embedding_size +
            obs_space["global_vars"].shape[0] +
            obs_space["last_action"].shape[0] +
            obs_space["last_reward"].shape[0]
        )

        # Encoder head
        self.encoder_head = make_nn_stack(
            input_size=all_embeddings_size,
            output_size=self._cfg.fc_size,
            hidden_sizes=[self._cfg.fc_size] * (self._cfg.fc_layers - 1),
            nonlinearity=nonlinearity(sf_cfg),
            layer_norm=self._cfg.fc_norm,
            use_skip=self._cfg.fc_skip,
        )

        self.encoder_head_out_size = self._cfg.fc_size

    def forward(self, obs_dict):
        grid_obs = obs_dict["grid_obs"]
        batch_size = grid_obs.size(0)
        embeddings = []
        # grid_features = grid_obs.permute(0, 2, 3, 1).view(batch_size, self._num_grid_features, np.prod(self._grid_shape))
        for fidx, grid_feature_name in enumerate(self._cfg.grid_feature_names):
            feature_ebd = self._embeddings[grid_feature_name](grid_obs[:, fidx, :, :].view(batch_size, -1))
            name_ebd = torch.tensor([self._embedding_name_ebd[grid_feature_name]]).to(feature_ebd.device)
            name_ebd =name_ebd.expand(batch_size, -1)
            embeddings.append(torch.cat([feature_ebd, name_ebd], dim=1))
        embeddings = torch.stack(embeddings, dim=1)

        emb_proj = self._embedding_proj(embeddings.reshape(-1, 1 + self._cfg.embedding_size))
        grid_embd = torch.sum(
            emb_proj.view(batch_size, -1, self._cfg.embedding_size),
            dim=1)

        # Additional features
        additional_features = torch.cat([
            obs_dict["global_vars"],
            obs_dict["last_action"].view(batch_size, -1),
            obs_dict["last_reward"].view(batch_size, -1)
        ], dim=1)

        all_obs = torch.cat([grid_embd, additional_features], dim=1)
        # Final encoding
        x = self.encoder_head(all_obs)
        return x.view(-1, self.encoder_head_out_size)

    def get_out_size(self) -> int:
        return self.encoder_head_out_size

    @classmethod
    def add_args(cls, parser):
        GridEncoder.add_args(parser)

class FeatureDictDecoder(MlpDecoder):
    pass


class FeatureDictAgent(SampleFactoryAgent):
    def encoder_cls(self):
        return FeatureDictEncoder

    def decoder_cls(self):
        return FeatureDictDecoder

