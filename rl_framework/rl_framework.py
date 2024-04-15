
from rl_framework.env_factory import EnvFactory
from rl_framework.agent_factory import AgentFactory

class RLFramework():
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
