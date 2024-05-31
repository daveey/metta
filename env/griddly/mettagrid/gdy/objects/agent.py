

from types import SimpleNamespace
from omegaconf import OmegaConf
from env.griddly.builder.object import GriddlyObject
import  env.griddly.mettagrid.gdy.sprites as sprites

def sprite_m(name: str):
    return f"oryx/oryx_tiny_galaxy/tg_sliced/tg_monsters/{name}.png"

class Agent(GriddlyObject):
    def __init__(self, game, cfg: OmegaConf):
        self.cfg = cfg

        super().__init__(
            game=game,
            name="agent",
            symbol="A",
            sprites=[
                sprites.monster("astronaut_u1"),
                sprites.monster("void_d1"),
                sprites.monster("beast_u1"),
                sprites.monster("void_d1"),
                sprites.monster("stalker_u1"),
            ],
            properties={
                "id": 0,
                "dir": 0,
                "energy": self.cfg.initial_energy,
                "shield": 0,
                "frozen": 0,
                "inv_r1": 0,
                "inv_r2": 0,
                "inv_r3": 0,
            },
        )


    def add_resource(self, actor, target):
        return [
        ]

    def on_init(self):
        return [
            self.prop("id").set("_playerId"),
        ]

