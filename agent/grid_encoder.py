

from sample_factory.model.encoder import Encoder
import torch

class GridEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        self._grid_obs_as_dict = False
        if cfg.agent_grid_obs_as_dict:
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

        self._griddly_max_features = cfg.agent_max_features

    # generate position encodings, shaped (1, 2, width, height)
    def _create_position_encodings(self):
        x = torch.linspace(-1, 1, self._grid_shape[0])
        y = torch.linspace(-1, 1, self._grid_shape[1])
        pos_x, pos_y = torch.meshgrid(x, y, indexing='xy')
        position_encodings = torch.stack((pos_x, pos_y), dim=-1)
        return position_encodings.unsqueeze(0).permute(0, 3, 1, 2)

    def _grid_obs(self, obs_dict):
        if self._grid_obs_as_dict:
            grid_obs = [ obs_dict[k] for k in self._grid_features ]
            return torch.cat(grid_obs, dim=1)
        else:
            return obs_dict["grid_obs"]

