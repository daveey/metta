from types import SimpleNamespace
from omegaconf import OmegaConf
from env.griddly.mettagrid.object.metta_object import MettaObject
import  env.griddly.mettagrid.util.sprite as sprite
from env.griddly.mettagrid.util.energy_helper import EnergyHelper
from env.griddly.mettagrid.util.inventory_helper import InventoryHelper

class Converter(MettaObject):
    Resources = ["r1", "r2", "r3"]

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
                sprite.item("pda_A"),
                sprite.item("pda_B"),
                sprite.item("pda_C"),
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


    def on_convert(self, ctx, resource: str, item_condition):
        ctx.require(self.usable(ctx))

        inv = InventoryHelper(ctx, ctx.actor)
        energy = EnergyHelper(ctx, ctx.actor)

        new_resource_idx = self.Resources.index(resource) + 1
        cmds = [
            energy.add(self.cfg.energy_output[resource], "used:converter"),
            ctx.global_var("game:max_steps").set("game:step"),
            ctx.global_var("game:max_steps").add(ctx.game.no_energy_steps),
        ]


        if new_resource_idx < len(self.Resources):
                cmds.append(inv.add(self.Resources[new_resource_idx], "converter"))

        if ctx.actor.object.cfg.energy_reward:
            cmds.append({"reward": self.cfg.energy_output[resource]})

        ctx.cmd(ctx.cond(item_condition, cmds))

        ctx.dst_cmd([
            ctx.target.state.set(ctx.target.object.States.cooldown),
            {"set_tile": ctx.target.state.val()},
            ctx.target.reset(ctx.target.object.cfg.cooldown)
        ])

