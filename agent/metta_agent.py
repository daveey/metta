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

        decoder_out_size: int = self._decoder.get_out_size()

        self._critic_linear = nn.Linear(decoder_out_size, 1)
        self._action_parameterization = self.get_action_parameterization(decoder_out_size)
        self._last_action_distribution = None

        self.apply(self.initialize_weights)

    def encode_observations(self, td: TensorDict):
        td["encoded_"] = self._encoder(obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self._core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        decoder_output = self._decoder(core_output)
        values = self._critic_linear(decoder_output).squeeze()

        result = TensorDict({"values": values}, batch_size=values.size(0))
        if values_only:
            return result

        action_distribution_params, self._last_action_distribution = self._action_parameterization(decoder_output)

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

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

    def get_action_parameterization(self, decoder_output_size: int):
        return ActionParameterizationDefault({}, decoder_output_size, self._action_space)

    def _maybe_sample_actions(self, sample_actions: bool, result: TensorDict) -> None:
        if sample_actions:
            # for non-trivial action spaces it is faster to do these together
            actions, result["log_prob_actions"] = sample_actions_log_probs(self._last_action_distribution)
            assert actions.dim() == 2  # TODO: remove this once we test everything
            result["actions"] = actions.squeeze(dim=1)

    def action_distribution(self):
        return self._last_action_distribution
