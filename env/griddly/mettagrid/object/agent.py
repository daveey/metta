

from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext
from env.griddly.mettagrid.util.energy_helper import EnergyHelper
from env.griddly.mettagrid.util.inventory_helper import InventoryHelper
from env.griddly.mettagrid.object.metta_object import MettaObject
import  env.griddly.mettagrid.util.sprite as sprite

def sprite_m(name: str):
    return f"oryx/oryx_tiny_galaxy/tg_sliced/tg_monsters/{name}.png"

class Agent(MettaObject):
    def __init__(self, game, cfg: OmegaConf):
        super().__init__(
            cfg=cfg,
            game=game,
            name="agent",
            symbol="A",
            sprites=[
                sprite.monster("astronaut_u1"),
                sprite.monster("void_d1"),
                sprite.monster("beast_u1"),
                sprite.monster("void_d1"),
                sprite.monster("stalker_u1"),
            ],
            properties={
                "id": 0,
                "dir": 0,
                "energy": cfg.initial_energy,
                "shield": 0,
                "frozen": 0,
                **InventoryHelper.Properties,
            },
        )
        self.register_action("init", self._init, initial=True)
        self.register_action("tick", self._tick, initial=True)
        self.register_action("unfreeze", self._unfreeze)

    def _init(self, ctx: BehaviorContext):
        ctx.cmd([
            ctx.actor.id.set("_playerId")
        ])

    def _tick(self, ctx: BehaviorContext):
        energy = EnergyHelper(ctx, ctx.actor)
        ctx.cmd([
            # turn off the shield if we run out of energy
            ctx.cond(ctx.actor.energy.lte(0), [
                ctx.global_var("game:agent:dead").incr(),
                {"remove": True }
            ]),

            energy.use(self.cfg.upkeep.time, "upkeep:time"),
            ctx.cond(ctx.actor.shield.eq(1),
                energy.use(self.cfg.upkeep.shield, "upkeep:shield")
            ),

            {"set_tile": 0},
            ctx.cond(ctx.actor.frozen.eq(1), {"set_tile": 1}),
            ctx.cond(ctx.actor.shield.eq(1), {"set_tile": 2}),

            ctx.player_var(f"debug:agent:energy").set(ctx.actor.energy.val()),
            ctx.player_var(f"debug:agent:inv_r1").set(ctx.actor.inv_r1.val()),
            ctx.actor.tick(delay=1)
        ])

    def _unfreeze(self, ctx: BehaviorContext):
        ctx.cmd([
            ctx.actor.frozen.set(0)
        ])

    def shield_on_cmds(self, ctx: BehaviorContext):
        return [
            ctx.actor.shield.set(1),
        ]

    def shield_off_cmds(self, ctx: BehaviorContext):
        return [
            ctx.actor.shield.set(0),
        ]

    def on_attack(self, ctx: BehaviorContext, damage: int):
        ctx.require([
            ctx.target.frozen.eq(0),
        ])

        energy = EnergyHelper(ctx, ctx.target)
        attacker_inv = InventoryHelper(ctx, ctx.actor)
        target_inv = InventoryHelper(ctx, ctx.target)

        frozen_cond = {
            "or": [ctx.target.shield.eq(0), ctx.target.energy.lt(damage)]
        }

        inv_add_cmds = []
        inv_rem_cmds = []
        for r in InventoryHelper.Items.values():
            inv_add_cmds.extend(attacker_inv.add(r, "loot", target_inv.item_var(r).val()))
            inv_rem_cmds.extend(target_inv.remove(r, "loot", target_inv.item_var(r).val()))

        ctx.cmd(ctx.cond(
            frozen_cond, [
                inv_add_cmds,
                inv_rem_cmds,
                ctx.player_var(f"stats:agent:attack:victory").incr(),
                ctx.target.frozen.set(1),
                ctx.player_var(f"stats:agent:defeated").incr(),
            ], [
                energy.use(damage, "shield:damage"),
            ]
        ))

        ctx.dst_cmd([
            ctx.cond(frozen_cond, [
                {"set_tile": 1},
                ctx.target.unfreeze(delay=self.cfg.freeze_duration),
            ])
        ])
