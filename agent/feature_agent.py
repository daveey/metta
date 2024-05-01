from __future__ import annotations

from matplotlib.pyplot import grid
import numpy as np
from sample_factory.model.encoder import Encoder
import torch

from sample_factory.model.decoder import MlpDecoder
from sample_factory.model.model_utils import nonlinearity

from agent.lib.feature_encoder import FeatureListEncoder
from agent.lib.normalizer import FeatureListNormalizer
from .lib.util import make_nn_stack
from agent.sample_factory_agent import SampleFactoryAgent

class FeatureEncoder(Encoder):

    def __init__(self, sf_cfg, obs_space, agent_cfg):
        super().__init__(sf_cfg)
        self._cfg = agent_cfg

        self._grid_features_names = []
        self._grid_obs_keys = []
        self._global_feature_names = []
        self._global_obs_keys = []
        self._grid_obs_shape = None

        for obs_name, feature_names in agent_cfg.feature_schema.items():
            assert len(feature_names) == obs_space[obs_name].shape[0], \
                "Number of features in schema does not match observation space"\
                f"for {obs_name}: {feature_names} vs {obs_space[obs_name].shape[0]}"
            if len(obs_space[obs_name].shape) == 1:
                print(f"Adding {obs_name} - {feature_names} as global feature set")
                self._global_feature_names.extend(feature_names)
                self._global_obs_keys.append(obs_name)
            elif len(obs_space[obs_name].shape) == 3:
                print(f"Adding {obs_name} - {feature_names} as grid feature set")
                if self._grid_obs_shape is None:
                    self._grid_obs_shape = obs_space[obs_name].shape[1:]
                else:
                    assert self._grid_obs_shape == obs_space[obs_name].shape[1:], \
                        "All grid observations must have the same shape"
                self._grid_features_names.extend(feature_names)
                self._grid_obs_keys.append(obs_name)

        self._global_normalizer = FeatureListNormalizer(self._global_feature_names)
        self._grid_normalizer = FeatureListNormalizer(
            self._grid_features_names, self._grid_obs_shape)

        self._global_encoder = FeatureListEncoder(
            self._cfg.globals_embedding,
            self._global_feature_names)

        ge_cfg = self._cfg.grid_embedding
        ge_cfg.input_dim = np.prod(self._grid_obs_shape)
        self._grid_encoder = FeatureListEncoder(
            ge_cfg, self._grid_features_names)

        # Encoder head
        self.encoder_head = make_nn_stack(
            input_size=self._cfg.grid_embedding.output_dim + self._cfg.globals_embedding.output_dim,
            output_size=self._cfg.fc_size,
            hidden_sizes=[self._cfg.fc_size] * (self._cfg.fc_layers - 1),
            nonlinearity=nonlinearity(sf_cfg),
            layer_norm=self._cfg.fc_norm,
            use_skip=self._cfg.fc_skip,
        )

        self.encoder_head_out_size = self._cfg.fc_size

    def forward(self, obs_dict):
        if len(self._global_obs_keys) >= 0:
            batch_size = obs_dict[self._global_obs_keys[0]].size(0)
        else:
            batch_size = obs_dict[self._grid_obs_keys[0]].size(0)

        global_obs = torch.concat([
            obs_dict[obs_key].view(batch_size, -1)
            for obs_key in self._global_obs_keys], dim=-1).unsqueeze(-1)

        grid_obs = torch.concat([
            obs_dict[obs_key].view(batch_size, -1, *self._grid_obs_shape)
            for obs_key in self._grid_obs_keys], dim=-1)

        if self._cfg.normalize_features:
            self._global_normalizer(global_obs)
            self._grid_normalizer(grid_obs)

        global_state = self._global_encoder(global_obs)
        grid_state = self._grid_encoder(grid_obs)

        # Final encoding
        x = self.encoder_head(torch.cat([global_state, grid_state], dim=1))
        return x

    def get_out_size(self) -> int:
        return self.encoder_head_out_size


class FeatureDecoder(MlpDecoder):
    pass


class FeatureAgent(SampleFactoryAgent):
    def encoder_cls(self):
        return FeatureEncoder

    def decoder_cls(self):
        return FeatureDecoder

