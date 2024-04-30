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

        self._grid_features = []
        self._grid_obs_keys = []
        self._global_features = []
        self._global_obs_keys = []
        grid_obs_shape = None


        for obs_name, feature_names in agent_cfg.feature_schema:
            if len(obs_space[obs_name].shape) == 1:
                print(f"Adding {obs_name} - {feature_names} as global feature set")
                self.global_features.extend(feature_names)
                self.global_obs_keys.append(obs_name)
            elif len(obs_space[obs_name].shape) == 3:
                print(f"Adding {obs_name} - {feature_names} as grid feature set")
                if grid_obs_shape is None:
                    grid_obs_shape = obs_space[obs_name].shape[1:]
                else:
                    assert grid_obs_shape == obs_space[obs_name].shape[1:], \
                        "All grid observations must have the same shape"
                self.grid_features.extend(feature_names)
                self.grid_obs_keys.append(obs_name)

        assert len(self._cfg.grid_feature_names) == grid_obs_shape[0], \
            f"Number of grid features in config ({len(self._cfg.grid_feature_names)}) " \
            f"does not match the number of grid features in the observation space ({obs_space['grid_obs'].shape[0]})"

        global_feature_names = self._cfg.global_feature_names + [
            "last_action_id", "last_action_val", "last_reward"
        ]
        self._grid_shape = obs_space["grid_obs"].shape[1:]

        self._global_normalizer = FeatureListNormalizer(global_feature_names)
        self._grid_normalizer = FeatureListNormalizer(
            self._cfg.grid_feature_names, self._grid_shape)

        self._global_encoder = FeatureListEncoder(
            self._cfg.globals_embedding,
            global_feature_names)

        ge_cfg = self._cfg.grid_embedding
        ge_cfg.input_dim = np.prod(self._grid_shape)
        self._grid_encoder = FeatureListEncoder(
            ge_cfg, self._cfg.grid_feature_names)

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
        global_obs = torch.concat([
            obs_dict["global_vars"],
            obs_dict["last_action"].view(-1, 2),
            obs_dict["last_reward"].view(-1, 1)], dim=-1).unsqueeze(-1)

        if self._cfg.normalize_features:
            self._global_normalizer(global_obs)
            self._grid_normalizer(obs_dict["grid_obs"])

        global_state = self._global_encoder(global_obs)
        grid_state = self._grid_encoder(obs_dict["grid_obs"])

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

