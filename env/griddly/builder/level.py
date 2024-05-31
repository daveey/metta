from omegaconf import OmegaConf


class LevelBuilder():
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg

    def build(self):
        raise NotImplementedError
