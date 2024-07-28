
import hydra
from tensordict import TensorDict
import pufferlib
import pufferlib.models
import pufferlib.pytorch
from omegaconf import OmegaConf
from pufferlib.emulation import PettingZooPufferEnv
from pufferlib.environment import PufferEnv
from torch import nn

from agent.metta_agent import MettaAgent


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class PufferAgentWrapper(nn.Module):
    def __init__(self, agent: MettaAgent, env: PettingZooPufferEnv):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        # xcxc
        self.atn_type = nn.Linear(agent.decoder_out_size(), env.action_space(1)[0].n)
        self.atn_param = nn.Linear(agent.decoder_out_size(), env.action_space(1)[1].n)
        self._agent = agent

    def forward(self, obs):
        x, _ = self.encode_observations(obs)
        return self.decode_actions(x, None)

    def encode_observations(self, flat_obs):
        obs = pufferlib.pytorch.nativize_tensor(flat_obs, self.dtype)
        td = TensorDict({"obs": obs})
        self._agent.encode_observations(td)
        return td["encoded_obs"], None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = [self.atn_type(flat_hidden), self.atn_param(flat_hidden)]
        value = self._agent._critic_linear(flat_hidden)
        return action, value

def make_policy(env: PufferEnv, cfg: OmegaConf):
    cfg.agent.observation_encoders.grid_obs.feature_names = env.env.unwrapped._gym_env.grid_features
    cfg.agent.observation_encoders.global_vars.feature_names = env.env.unwrapped._gym_env.global_features
    agent = hydra.utils.instantiate(
        cfg.agent, env.env.observation_space(0),
        env.env.action_space(0), _recursive_=False)
    puffer_agent = PufferAgentWrapper(agent, env)

    if cfg.agent.core.rnn_num_layers > 0:
        puffer_agent = Recurrent(
            env, puffer_agent, input_size=cfg.agent.fc.output_dim,
            hidden_size=cfg.agent.core.rnn_size,
            num_layers=cfg.agent.core.rnn_num_layers
        )
        puffer_agent = pufferlib.frameworks.cleanrl.RecurrentPolicy(puffer_agent)
    else:
        puffer_agent = pufferlib.frameworks.cleanrl.Policy(puffer_agent)

    return puffer_agent.to(cfg.framework.pufferlib.device)
