

from __future__ import annotations

from sample_factory.algo.utils.tensor_dict import TensorDict


from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config

from agent.metta_agent import MettaAgent

from typing import Dict

from torch import Tensor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.typing import Config

class SampleFactoryAgentWrapper(ActorCritic):
    def __init__(self, agent: MettaAgent, obs_space, action_space):
        super().__init__(obs_space, action_space, AttrDict({
            "normalize_returns": True,
            "normalize_input": False,
            "obs_subtract_mean": 0.0,
            "obs_scale": 1.0,

        }))
        self.encoders = [agent.encoder]

        self._agent = agent

    def forward_head(self, obs_dict: Dict[str, Tensor]) -> Tensor:
        return self._agent.forward_head(obs_dict)

    def forward_core(self, head_output, rnn_states):
        return self._agent.forward_core(head_output, rnn_states)

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        return self._agent.forward_tail(core_output, values_only, sample_actions)

    def forward(self, obs_dict, rnn_states, values_only: bool = False) -> TensorDict:
        return self._agent.forward(obs_dict, rnn_states, values_only)

    def action_distribution(self):
        return self._agent.action_distribution()

    def aux_loss(self, obs_dict, rnn_states):
        return self._agent.aux_loss(obs_dict, rnn_states)
