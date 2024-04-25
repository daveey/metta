from __future__ import annotations
import glob
from types import SimpleNamespace


from cv2 import norm
from matplotlib.pyplot import grid
import numpy as np
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
from sample_factory.model.encoder import Encoder
from sample_factory.utils.normalize import ObservationNormalizer
from torch import nn
import torch
import gymnasium as gym

from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity
from agent.grid_encoder import GridEncoder
from .util import make_nn_stack, embed_strings, sinusoidal_position_embeddings
from agent.sample_factory_agent import SampleFactoryAgent

class FeatureDictEncoder(Encoder):

    def __init__(self, sf_cfg, obs_space, agent_cfg):
        super().__init__(sf_cfg)
        self._cfg = agent_cfg
        self._grid_feature_names = self._cfg.grid_feature_names
        self._global_feature_names = self._cfg.global_feature_names + [
            "last_action_id", "last_action_val", "last_reward"
        ]

        assert len(self._grid_feature_names) == obs_space["grid_obs"].shape[0], \
            f"Number of grid features in config ({len(self._grid_feature_names)}) " \
            f"does not match the number of grid features in the observation space ({obs_space['grid_obs'].shape[0]})"

        self._grid_shape = obs_space["grid_obs"].shape[1:]
        self._num_grid_features = obs_space["grid_obs"].shape[0]

        # Generate the global feature id embeddings
        self._global_meta_embs = embed_strings(self._global_feature_names, self._cfg.feature_id_embedding_dim)

        # Generate the feature id embeddings
        self._grid_feature_ids = embed_strings(self._grid_feature_names, self._cfg.feature_id_embedding_dim)
        self._grid_meta_embs = self._grid_feature_ids\
            .view(-1, self._cfg.feature_id_embedding_dim)\

        # If we are using position embeddings, add them to the grid embeddings
        grid_obs_size = np.prod(self._grid_shape) + self._grid_meta_embs.size(-1)
        if self._cfg.add_position_embeddings:
            pos_embs = sinusoidal_position_embeddings(*self._grid_shape, self._cfg.position_embedding_dim)
            pos_embs = pos_embs.expand(self._grid_meta_embs.size(0), *self._grid_shape, -1)
            self._grid_meta_embs = torch.cat([
                self._grid_meta_embs.unsqueeze(1).unsqueeze(1).expand(-1, *self._grid_shape, -1),
                pos_embs.expand(self._grid_meta_embs.size(0), -1 -1, -1)
            ], dim=-1)
            grid_obs_size = 1 + self._grid_meta_embs.size(-1)

        # Create the observation normalizers
        self._obs_norm_dict = nn.ModuleDict({
            **{
                k: RunningMeanStdInPlace(self._grid_shape)
                for k in self._grid_feature_names
            },
            **{
                k: RunningMeanStdInPlace((1,))
                for k in self._global_feature_names
            },
        })
        self._grid_norms = [self._obs_norm_dict[k] for k in self._grid_feature_names]
        self._globals_norms = [self._obs_norm_dict[k] for k in self._global_feature_names]

        self._global_net = make_nn_stack(
            input_size=self._global_meta_embs.size(-1) + 1,
            output_size=self._cfg.globals_emb_size,
            hidden_sizes=[self._cfg.globals_emb_size] * (self._cfg.globals_emb_layers - 1),
            nonlinearity=nonlinearity(sf_cfg),
        )

        # Every grid feature takes WxH + feature_id + global_emb_size
        self._grid_feature_net = make_nn_stack(
            input_size=grid_obs_size,
            output_size=self._cfg.grid_emb_size,
            hidden_sizes=[self._cfg.grid_emb_size] * (self._cfg.grid_emb_layers - 1),
            nonlinearity=nonlinearity(sf_cfg),
        )

        # Encoder head
        self.encoder_head = make_nn_stack(
            input_size=self._cfg.grid_emb_size + self._cfg.globals_emb_size,
            output_size=self._cfg.fc_size,
            hidden_sizes=[self._cfg.fc_size] * (self._cfg.fc_layers - 1),
            nonlinearity=nonlinearity(sf_cfg),
            layer_norm=self._cfg.fc_norm,
            use_skip=self._cfg.fc_skip,
        )

        self.encoder_head_out_size = self._cfg.fc_size

    def forward(self, obs_dict):
        # obs_dict = ObservationNormalizer._clone_tensordict(obs_dict)
        self._global_meta_embs = self._global_meta_embs.to(obs_dict["global_vars"].device)
        self._grid_meta_embs = self._grid_meta_embs.to(obs_dict["grid_obs"].device)

        global_vars = obs_dict["global_vars"].unsqueeze(-1)
        batch_size = obs_dict["grid_obs"].size(0)

        # Normalize global features
        if self._cfg.normalize_features:
            with torch.no_grad():
                for fidx, norm in enumerate(self._globals_norms):
                    norm(global_vars[:, fidx, :])

        # Embed every (feature_id,value), then sum them up
        global_obs = torch.concat(
            [obs_dict["global_vars"],
             obs_dict["last_action"],
             obs_dict["last_reward"]], dim=-1)
        global_obs = torch.cat([
            self._global_meta_embs.unsqueeze(0).expand(batch_size, -1, -1),
            global_obs.unsqueeze(-1)], dim=-1)

        globals_embed = self._global_net(global_obs.view(-1, global_obs.size(-1)))
        globals_embed = globals_embed.view(batch_size, -1, self._cfg.globals_emb_size)
        global_state = torch.sum(globals_embed, dim=1)

        # Normalize grid features
        if self._cfg.normalize_features:
            with torch.no_grad():
                for fidx, norm in enumerate(self._grid_norms):
                    norm(obs_dict["grid_obs"][:, fidx, :, :])

        # Combine the grid embeddings with the grid observations
        if self._cfg.add_position_embeddings:
            grid_obs = torch.cat([
                self._grid_meta_embs.expand(batch_size, -1, -1, -1, -1),
                obs_dict["grid_obs"].unsqueeze(-1)], dim=-1)
        else:
            grid_obs = torch.cat([
                self._grid_meta_embs.expand(batch_size, -1, -1),
                obs_dict["grid_obs"].view(batch_size, self._num_grid_features, -1)
            ], dim=-1)

        grid_obs = grid_obs.view(-1, self._grid_feature_net[0].in_features)
        grid_embs = self._grid_feature_net(grid_obs).view(batch_size, -1, self._cfg.grid_emb_size)
        grid_state = torch.sum(grid_embs, dim=1)

        # Final encoding
        x = self.encoder_head(torch.cat([global_state, grid_state], dim=1))
        return x

    def get_out_size(self) -> int:
        return self.encoder_head_out_size


class FeatureDictDecoder(MlpDecoder):
    pass


class FeatureDictAgent(SampleFactoryAgent):
    def encoder_cls(self):
        return FeatureDictEncoder

    def decoder_cls(self):
        return FeatureDictDecoder

