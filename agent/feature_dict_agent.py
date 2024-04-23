from __future__ import annotations
import glob


from cv2 import norm
import numpy as np
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
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

        self._global_net = make_nn_stack(
            input_size=obs_space["global_vars"].shape[0],
            output_size=self._cfg.globals_emb_size,
            hidden_sizes=[self._cfg.globals_emb_size] * (self._cfg.globals_emb_layers - 1),
            nonlinearity=nonlinearity(sf_cfg),
        )

        self._grid_shape = obs_space["grid_obs"].shape[1:]
        self._num_grid_features = obs_space["grid_obs"].shape[0]

        self._grid_norm_dict = nn.ModuleDict({
            k: RunningMeanStdInPlace(self._grid_shape)
            for k in self._cfg.grid_feature_names
        })
        self._grid_norm = [self._grid_norm_dict[k] for k in self._cfg.grid_feature_names]

        # Every grid feature takes WxH + feature_id + global_emb_size
        self._grid_feature_net = make_nn_stack(
            input_size=1 + np.prod(self._grid_shape) + self._cfg.globals_emb_size,
            output_size=self._cfg.grid_emb_size,
            hidden_sizes=[self._cfg.grid_emb_size] * (self._cfg.grid_emb_layers - 1),
            nonlinearity=nonlinearity(sf_cfg),
        )
        self._grid_feature_ids = []
        for grid_feature_name in self._cfg.grid_feature_names:
            self._grid_feature_ids.append(
                  hash(grid_feature_name) % 100000 / 100000.0)
        self._grid_feature_ids = torch.tensor(self._grid_feature_ids).to(torch.float32)
        self._grid_feature_ids = self._grid_feature_ids.unsqueeze(0).unsqueeze(-1)

        # all_embeddings_size = (
        #     obs_space["last_action"].shape[0] +
        #     obs_space["last_reward"].shape[0]
        # )

        # Encoder head
        self.encoder_head = make_nn_stack(
            input_size=self._cfg.grid_emb_size,
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

        for fidx, norm in enumerate(self._grid_norm):
            norm(obs_dict["grid_obs"][:, fidx, :, :])

        grid_obs = grid_obs.view(batch_size, -1, np.prod(self._grid_shape))

        globals_embed = self._global_net(obs_dict["global_vars"].view(batch_size, 1, -1))
        globals_embed = globals_embed.expand(-1, grid_obs.size(1), -1)

        fids = self._grid_feature_ids.to(grid_obs.device).expand(batch_size, -1, -1)

        grid_obs = torch.cat([fids, globals_embed, grid_obs], dim=2)

        grid_embed = self._grid_feature_net(grid_obs)


        # # Additional features
        # additional_features = torch.cat([
        #     obs_dict["global_vars"],
        #     obs_dict["last_action"].view(batch_size, -1),
        #     obs_dict["last_reward"].view(batch_size, -1)
        # ], dim=1)

        # Final encoding
        x = self.encoder_head(torch.sum(grid_embed, dim=1))
        return x

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

