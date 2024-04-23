

class RLFramework():
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
