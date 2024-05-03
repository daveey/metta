from __future__ import annotations

from typing import Dict

import hydra
from omegaconf import OmegaConf
from sample_factory.model.action_parameterization import ActionParameterizationDefault
from sample_factory.model.core import ModelCoreRNN
from sample_factory.model.decoder import MlpDecoder
from sample_factory.utils.typing import ActionSpace, ObsSpace
from torch import Tensor
from sample_factory.algo.utils.action_distributions import is_continuous_action_space, sample_actions_log_probs

from tensordict import TensorDict
from torch import Tensor, nn
import torch
class MettaAgentInterface():
    def forward_head(self, obs_dict: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward_core(self, head_output, rnn_states):
        raise NotImplementedError()

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        raise NotImplementedError()

    def forward(self, obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        raise NotImplementedError()

    def aux_loss(self, obs_dict, rnn_states):
        raise NotImplementedError()

class MettaAgent(nn.Module, MettaAgentInterface):
    def __init__(
        self,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        **cfg
    ):
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self.observation_space = obs_space
        self.action_space = action_space

        self.encoder = hydra.utils.instantiate(cfg.encoder, obs_space)

        self.core = ModelCoreRNN(cfg.core, self.encoder.get_out_size())

        self.decoder = hydra.utils.instantiate(
            cfg.decoder,
            self.core.get_out_size())

        decoder_out_size: int = self.decoder.get_out_size()

        self.critic_linear = nn.Linear(decoder_out_size, 1)
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)
        self.last_action_distribution = None

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict({"values": values})
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output)

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
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

        # elif self.cfg.policy_initialization == "xavier_uniform":
        #     if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
        #         nn.init.xavier_uniform_(layer.weight.data, gain=gain)
        #     else:
        #         pass
        # elif self.cfg.policy_initialization == "torch_default":
        #     # do nothing
        #     pass

    def get_action_parameterization(self, decoder_output_size: int):
        return ActionParameterizationDefault({}, decoder_output_size, self.action_space)

    def _maybe_sample_actions(self, sample_actions: bool, result: TensorDict) -> None:
        if sample_actions:
            # for non-trivial action spaces it is faster to do these together
            actions, result["log_prob_actions"] = sample_actions_log_probs(self.last_action_distribution)
            assert actions.dim() == 2  # TODO: remove this once we test everything
            result["actions"] = actions.squeeze(dim=1)

    def action_distribution(self):
        return self.last_action_distribution
