from __future__ import annotations

from torch import nn
import torch

from .util import make_nn_stack, embed_strings

class FeatureListEncoder(nn.Module):
    def __init__(self, cfg, feature_names):
        super().__init__()
        self._cfg = cfg

        self._feature_names = feature_names
        self._num_features = len(feature_names)

        self._labels_emb = embed_strings(self._feature_names, cfg.label_dim)
        self._labels_emb = self._labels_emb.unsqueeze(0)

        self._embedding_net = make_nn_stack(
            input_size=cfg.label_dim + cfg.input_dim,
            output_size=cfg.output_dim,
            hidden_sizes=[cfg.output_dim] * (cfg.layers - 1),
            nonlinearity=nn.ELU(),
        )

    def forward(self, obs):
        batch_size = obs.size(0)
        self._labels_emb = self._labels_emb.to(obs.device)

        obs = obs.view(-1, self._num_features, self._cfg.input_dim)
        obs = torch.cat([
            self._labels_emb.expand(batch_size, -1, -1),
            obs
        ], dim=-1)
        obs = obs.view(-1, self._embedding_net[0].in_features)

        embs = self._embedding_net(obs).view(batch_size, -1, self._cfg.output_dim)
        return torch.sum(embs, dim=1)
