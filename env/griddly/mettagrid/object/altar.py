from types import SimpleNamespace
from omegaconf import OmegaConf
from env.griddly.mettagrid.object.metta_object import MettaObject
import  env.griddly.mettagrid.util.sprite as sprite

class Altar(MettaObject):
    def __init__(self, game, cfg: OmegaConf):

        self.States = SimpleNamespace(
            ready = 0,
            cooldown = 1
        )

        super().__init__(
            cfg=cfg,
            game=game,
            name = "altar",
            symbol = "a",
            sprites=[
                sprite.item("heart_clear"),
                sprite.item("heart_full"),
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

