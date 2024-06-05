from types import SimpleNamespace
from omegaconf import OmegaConf
from env.griddly.mettagrid.gdy.objects.metta_object import MettaObject
import  env.griddly.mettagrid.gdy.sprites as sprites
from env.griddly.mettagrid.gdy.util.energy_helper import EnergyHelper
from env.griddly.mettagrid.gdy.util.inventory_helper import InventoryHelper

class Converter(MettaObject):
    def __init__(self, game, cfg: OmegaConf):

        self.States = SimpleNamespace(
            ready = 0,
            cooldown = 1
        )

        super().__init__(
            cfg=cfg,
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
        inv = InventoryHelper(ctx, ctx.actor)
        energy = EnergyHelper(ctx, ctx.actor)
        ctx.require([
            inv.has_item("r1")
        ])

        ctx.cmd([
            inv.remove("r1", "converter"),
            inv.add("r2", "converter"),
            energy.add(self.cfg.energy_output, "used:converter"),
        ])

        ctx.dst_cmd([
            ctx.target.state.set(ctx.target.object.States.cooldown),
            {"set_tile": ctx.target.state.val()},
            ctx.target.reset(ctx.target.object.cfg.cooldown)
        ])

