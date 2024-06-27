

from __future__ import annotations
from xml.dom.expatbuilder import theDOMImplementation

from sample_factory.model.core import ModelCoreRNN
from tensordict import TensorDict


from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.model_utils import model_device
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
import torch

from agent.metta_agent import MettaAgent

from typing import Dict

from torch import Tensor


class SampleFactoryAgentWrapper(ActorCritic):
    def __init__(self, agent: MettaAgent, obs_space, action_space):
        super().__init__(obs_space, action_space, AttrDict({
            "normalize_returns": True,
            "normalize_input": False,
            "obs_subtract_mean": 0.0,
            "obs_scale": 1.0,

        }))
        self.agent = agent
        decoder_out_size = self.agent._decoder.get_out_size()
        self._core = ModelCoreRNN(cfg.core, cfg.fc.output_dim)
        self._action_parameterization = self.get_action_parameterization(decoder_out_size)
        self._last_action_distribution = None

    def forward_head(self, obs_dict: Dict[str, Tensor]) -> Tensor:
        return self.agent.encode_observations(
            TensorDict(obs=obs_dict))["encoded_obs"]

    def forward_core(self, head_output, rnn_states):
        x, new_rnn_states = self._core(head_output, rnn_states)
        return x, new_rnn_states
        return self.agent.forward_core(
            TensorDict(
                encoded_obs=head_output,
                rnn_states=rnn_states))["core_output"]

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        td = TensorDict(core_output=core_output)
        self.agent.forward_tail(td, values_only)
        if not values_only:
            action_distribution_params, self._last_action_distribution = self._action_parameterization(td["decoder_output"])
            # `action_logits` is not the best name here, better would be "action distribution parameters"
            td["action_logits"] = action_distribution_params
            self._maybe_sample_actions(sample_actions, td)
        return td

    def forward(self, obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        return self.agent.forward(obs_dict, rnn_states, values_only)

    def action_distribution(self):
        return self._last_action_distribution

    def _maybe_sample_actions(self, sample_actions: bool, result: TensorDict) -> None:
        if sample_actions:
            # for non-trivial action spaces it is faster to do these together
            actions, result["log_prob_actions"] = sample_actions_log_probs(self._last_action_distribution)
            assert actions.dim() == 2  # TODO: remove this once we test everything
            result["actions"] = actions.squeeze(dim=1)


    def aux_loss(self, obs_dict, rnn_states):
        return self.agent.aux_loss(obs_dict, rnn_states)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32

    def get_action_parameterization(self, decoder_output_size: int):
        return ActionParameterizationDefault({}, decoder_output_size, self._action_space)

