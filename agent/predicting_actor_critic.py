

from __future__ import annotations

from sample_factory.algo.utils.tensor_dict import TensorDict

import numpy as np
from torch import nn

from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.utils.typing import Config
import torch
from sample_factory.algo.utils.context import global_model_factory


class PredictingActorCritic(ActorCriticSharedWeights):
    def __init__(self, model_factory, obs_space, action_space, cfg: Config):
        super().__init__(model_factory, obs_space, action_space, cfg)
        self.obs_size = np.prod(obs_space["obs"].shape)
        self.obs_predictor = nn.Sequential(
            nn.Linear(self.encoder.get_out_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.obs_size),
            nn.Sigmoid()  # Sigmoid to constrain output between 0 and 1
        )

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        result = super().forward(normalized_obs_dict, rnn_states, values_only)

        # compute the prediction error of observations
        # by reconstructing the observations from the rnn states
        # and pass it through as the last action
        if not values_only:
            batch_size = rnn_states.shape[0]
            obs = normalized_obs_dict['obs'].view(batch_size, -1)
            obs_pred = self.obs_predictor(rnn_states)
            error = torch.mean((obs - obs_pred) ** 2, dim=1)
            result["actions"][:,1] = error

        return result

def make_actor_critic_func(cfg, obs_space, action_space):
    return PredictingActorCritic(
        global_model_factory(), obs_space, action_space, cfg)
