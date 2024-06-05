from omegaconf import OmegaConf

from env.griddly.mettagrid.object.metta_object import MettaObject


class Wall(MettaObject):
    def __init__(self, game, cfg: OmegaConf):
        super().__init__(
            cfg=cfg,
            game=game,
            name = "wall",
            symbol = "W",
            sprites = [ f"oryx/oryx_fantasy/wall2-{i}.png" for i in range(16) ],
            tiling_mode="WALL_16",
            properties={}
        )
