from types import SimpleNamespace
from omegaconf import OmegaConf
from env.griddly.builder.action import GriddlyAction, GriddlyInternalAction
from env.griddly.builder.object import GriddlyObject
import env.griddly.builder.commands as cmd

class Generator(GriddlyObject):
    def __init__(self, game, cfg: OmegaConf):
        self.cfg = cfg
        self.States = SimpleNamespace(
            ready = 0,
            cooldown = 1,
            empty = 2
        )

        super().__init__(
            game = game,
            name = "generator",
            symbol = "g",
            sprites = [
                f"oryx/oryx_fantasy/ore-1.png",
                f"oryx/oryx_fantasy/ore-2.png",
                f"oryx/oryx_fantasy/ore-0.png",
            ],
            properties = {
                "state": self.States.ready,
                "amount": cfg.initial_resources,
            }
        )

        self.cooldown = cfg.cooldown
        self.register_action("reset", self.on_reset)

    def on_reset(self, ctx):
        ctx.cmd([
            ctx.target.state.set(ctx.target.object.States.ready),
            {"set_tile": ctx.target.state.val()},
        ])

    def on_use(self, ctx):
        ctx.require([
            ctx.actor.inv_r1.lt(ctx.actor.object.cfg.max_inventory),
            ctx.target.amount.gt(0)
        ])

        ctx.cmd(ctx.actor.inv_r1.incr())

        ctx.dst_cmd([
            ctx.target.amount.decr(),
            ctx.cond(ctx.target.amount.gt(0), [
                ctx.target.state.set(ctx.target.object.States.cooldown),
                ctx.target.reset(ctx.target.object.cfg.cooldown)
            ], [
                ctx.target.state.set(ctx.target.object.States.empty)
            ]),
            {"set_tile": ctx.target.state.val()},
        ])
