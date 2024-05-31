from types import SimpleNamespace
from omegaconf import OmegaConf
from env.griddly.builder.object import GriddlyObject
import  env.griddly.mettagrid.gdy.sprites as sprites

class Converter(GriddlyObject):
    def __init__(self, game, cfg: OmegaConf):
        self.cfg = cfg

        self.States = SimpleNamespace(
            ready = 0,
            cooldown = 1
        )

        super().__init__(
            game=game,
            name = "converter",
            symbol = "c",
            sprites=[
                sprites.item("pda_A"),
                sprites.item("pda_B"),
                sprites.item("pda_C"),
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
        actor_cfg = ctx.actor.object.cfg
        ctx.require([ctx.actor.inv_r1.gt(0)])
        ctx.cmd([
            ctx.actor.inv_r1.decr(),
            ctx.actor.inv_r2.incr(),
            ctx.cond(ctx.actor.inv_r2.gt(actor_cfg.max_inventory), [
                ctx.actor.inv_r2.set(actor_cfg.max_inventory)
            ], []),
            ctx.actor.energy.add(self.cfg.energy_output),
            {"reward": self.cfg.energy_output},
            ctx.cond(ctx.actor.energy.gt(actor_cfg.max_energy), [
                ctx.actor.energy.set(actor_cfg.max_energy)
            ], [])

        ])
        ctx.dst_cmd([
            ctx.target.state.set(ctx.target.object.States.cooldown),
            {"set_tile": ctx.target.state.val()},
            ctx.target.reset(ctx.target.object.cfg.cooldown)
        ])

