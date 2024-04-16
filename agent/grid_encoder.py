

import json
from numpy import pad
from sample_factory.model.encoder import Encoder
import torch
from omegaconf import OmegaConf

class GridEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self._cfg = OmegaConf.create(json.loads(cfg.agent_cfg))

        self._grid_obs_as_dict = False
        if self._grid_obs_as_dict:
            grid_obs_spaces = [
                v for k, v in obs_space.items() if v.shape == (1, 11, 11)
            ]
            self._grid_features = [
                k for k, v in obs_space.items() if v.shape == (1, 11, 11)
            ]
            self._num_grid_features = len(self._grid_features)
            self._grid_shape = grid_obs_spaces[0].shape[1:]
        else:
            grid_obs_space = obs_space["grid_obs"]
            self._grid_features = ["grid_obs"]
            self._num_grid_features = grid_obs_space.shape[0]
            self._grid_shape = grid_obs_space.shape[1:3]

        if self._cfg.get("add_position", False):
            self._num_grid_features += 2
            self._position_encodings = self._create_position_encodings()

        if self._cfg.get("num_grid_features"):
            self._features_padding = torch.zeros(
                1, # batch
                self._cfg.get("num_grid_features") - self._num_grid_features,
                *self._grid_shape)
            self._num_grid_features = self._cfg.get("num_grid_features")

        self._shuffle_features = self._cfg.get("shuffle_features", False)
        self._shuffle_features_table = None

    def _grid_obs(self, obs_dict):
        if self._grid_obs_as_dict:
            grid_obs = [ obs_dict[k] for k in self._grid_features ]
            grid_obs = torch.cat(grid_obs, dim=1)
        else:
            grid_obs = obs_dict["grid_obs"]

        batch_size = grid_obs.size(0)

        # Add padding
        if self._features_padding is not None:
            padding = self._features_padding.expand(batch_size, -1, -1, -1)
            padding = padding.to(grid_obs.device)
            grid_obs = torch.cat([grid_obs, padding], dim=1)

        # Add position encodings
        if self._position_encodings is not None:
            pos = self._position_encodings.expand(batch_size, -1, -1, -1)
            pos = pos.to(grid_obs.device)
            grid_obs = torch.cat([pos, grid_obs], dim=1)

        # shuffle the grid_obs to make sure the order of the features is not important
        if self._shuffle_features:
            if self._shuffle_features_table is None or torch.rand(1) < self._cfg.get("shuffle_features_update_prob", 0.00001):
                self._shuffle_features_table = torch.stack([
                    torch.randperm(self._num_grid_features)
                    for _ in range(self._cfg.get("shuffle_features_size"))
                ]).to(obs_dict["rollout_info"].device)
                print("Shuffle features table updated")

            idxs = torch.fmod(
                torch.sum(obs_dict["rollout_info"] *
                torch.tensor([1, 100, 1]).to(grid_obs.device), dim=1),
                len(self._shuffle_features_table)).int()
            perms = self._shuffle_features_table[idxs]
            grid_obs = grid_obs[torch.arange(batch_size).unsqueeze(1), perms, :, :]

        return grid_obs

    # generate position encodings, shaped (1, 2, width, height)
    def _create_position_encodings(self):
        x = torch.linspace(-1, 1, self._grid_shape[0])
        y = torch.linspace(-1, 1, self._grid_shape[1])
        pos_x, pos_y = torch.meshgrid(x, y, indexing='xy')
        position_encodings = torch.stack((pos_x, pos_y), dim=-1)
        return position_encodings.unsqueeze(0).permute(0, 3, 1, 2)
