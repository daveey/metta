from types import SimpleNamespace
from omegaconf import OmegaConf
from env.griddly.builder.object import GriddlyObject
import  env.griddly.mettagrid.gdy.sprites as sprites

class Altar(GriddlyObject):
    def __init__(self, game, cfg: OmegaConf):
        self.cfg = cfg

        self.States = SimpleNamespace(
            ready = 0,
            cooldown = 1
        )

        super().__init__(
            game=game,
            name = "altar",
            symbol = "a",
            sprites=[
                sprites.item("heart_clear"),
                sprites.item("heart_full"),
            ],
            properties={
                "state": self.States.ready,
            }
        )

        self.register_action("reset", self.on_reset)

    def on_reset(self, ctx):
        ctx.cmd([
            ctx.target.state.set(ctx.target.object.States.ready),
            {"set_tile": ctx.target.state.val()},
        ])


    def on_use(self, ctx):
        ctx.cmd([
            {"reward": 1},
        ])
        ctx.dst_cmd([
            ctx.target.state.set(ctx.target.object.States.cooldown),
            {"set_tile": ctx.target.state.val()},
            ctx.target.reset(ctx.target.object.cfg.cooldown)
        ])

