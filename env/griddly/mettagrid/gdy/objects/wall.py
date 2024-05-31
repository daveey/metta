from omegaconf import OmegaConf
from env.griddly.builder.object import GriddlyObject


class Wall(GriddlyObject):
    def __init__(self, game, cfg: OmegaConf):
        super().__init__(
            game=game,
            name = "wall",
            symbol = "W",
            sprites = [ f"oryx/oryx_fantasy/wall2-{i}.png" for i in range(16) ],
            tiling_mode="WALL_16"
        )
