

from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext
from env.griddly.builder.object import GriddlyObject
from env.griddly.mettagrid.util.energy_helper import EnergyHelper
from env.griddly.mettagrid.util.inventory_helper import InventoryHelper
from env.griddly.mettagrid.object.metta_object import MettaObject
import  env.griddly.mettagrid.util.sprite as sprite

class Clock(GriddlyObject):
    def __init__(self, game):
        super().__init__(
            game=game,
            name="clock",
            symbol="q",
            sprites=[
                sprite.sprite2d("diamond"),
            ],
            properties={
            },
        )
        self.register_action("init", self._init, initial=True)
        self.register_action("tick", self._tick, initial=True)

    def _init(self, ctx: BehaviorContext):
        ctx.cmd([
            ctx.global_var("game:step").set(0),
        ])

    def _tick(self, ctx: BehaviorContext):
        ctx.cmd([
            ctx.global_var("game:step").incr(),
            ctx.actor.tick(delay=1)
        ])
