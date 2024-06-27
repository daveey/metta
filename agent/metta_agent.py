from __future__ import annotations

from typing import Dict

import hydra
from omegaconf import OmegaConf
from sample_factory.model.action_parameterization import ActionParameterizationDefault
from sample_factory.model.core import ModelCoreRNN
from sample_factory.utils.typing import ActionSpace, ObsSpace
from torch import Tensor
from sample_factory.algo.utils.action_distributions import sample_actions_log_probs

from tensordict import TensorDict
from torch import Tensor, nn
import torch

from agent.agent_interface import MettaAgentInterface
from agent.feature_encoder import MultiFeatureSetEncoder

class MettaAgent(nn.Module, MettaAgentInterface):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self._cfg = cfg
        self._observation_space = obs_space
        self._action_space = action_space

        self._encoder = MultiFeatureSetEncoder(
            obs_space,
            cfg.observation_encoders,
            cfg.fc.layers,
            cfg.fc.output_dim
        )
        self._core = ModelCoreRNN(cfg.core, cfg.fc.output_dim)

        self._decoder = hydra.utils.instantiate(
            cfg.decoder,
            self._core.get_out_size())

        self._critic_linear = nn.Linear(self.decoder_out_size, 1)

        self.apply(self.initialize_weights)

    @property
    def decoder_out_size(self):
        return self._decoder.get_out_size()

    @property
    def core_out_size(self):
        return self._cfg.core.rnn_size

    def encode_observations(self, td: TensorDict):
        td["encoded_obs"] = self._encoder(td["obs"])

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self._core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, td: TensorDict):
        td["decoder_output"] = self._decoder(td["core_output"])
        td["values"] = self._critic_linear(td["decoder_output"]).squeeze()

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.encode_observations(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result

    def aux_loss(self, normalized_obs_dict, rnn_states):
        raise NotImplementedError()

    def initialize_weights(self, layer):
        gain = 1.0

        if hasattr(layer, "bias") and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
            nn.init.orthogonal_(layer.weight.data, gain=gain)
        else:
            # LSTMs and GRUs initialize themselves
            # should we use orthogonal/xavier for LSTM cells as well?
            # I never noticed much difference between different initialization schemes, and here it seems safer to
            # go with default initialization,
            pass
