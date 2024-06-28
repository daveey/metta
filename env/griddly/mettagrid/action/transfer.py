from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyAction, GriddlyActionInput
from env.griddly.builder.game_builder import GriddlyGameBuilder
from env.griddly.mettagrid.action.metta_action import MettaActionBehavior
from env.griddly.mettagrid.object.agent import Agent
from env.griddly.mettagrid.object.converter import Converter
from env.griddly.mettagrid.util.inventory_helper import InventoryHelper

class Transfer(GriddlyAction):
    def __init__(
            self,
            game: GriddlyGameBuilder,
            cfg: OmegaConf):

        super().__init__(
            name="transfer",
            game=game,
            inputs=[
                GriddlyActionInput(dest=[0, -1], metadata={"item": id}, description=f"Drop Inv {id}")
                for id in InventoryHelper.Items.keys() ]
        )
        self.cfg = cfg

        self.add_behaviour(MettaActionBehavior("agent", "agent", cfg, self.gift))
        self.add_behaviour(MettaActionBehavior("agent", "_empty", cfg, self.drop))
        self.add_behaviour(MettaActionBehavior("agent", "converter", cfg, self.convert))

    def drop(self, ctx: BehaviorContext):
        self._transfer(ctx, InventoryHelper(ctx, ctx.actor), None, "drop")

    def gift(self, ctx: BehaviorContext):
        self._transfer(ctx, InventoryHelper(ctx, ctx.actor), ctx.target, "gift")

    def convert(self, ctx: BehaviorContext):
        self._transfer(ctx, InventoryHelper(ctx, ctx.actor), ctx.target, "convert")

    def _transfer(self, ctx: BehaviorContext, actor_inv, target, reason: str):
        prereqs = []

        for item_id, item_name in InventoryHelper.Items.items():
            cnd = {"and": [
                ctx.metadata("item").eq(item_id),
                actor_inv.has_item(item_name),
            ]}
            prereqs.append(cnd)

            ctx.cmd(ctx.cond(cnd, [
                *actor_inv.remove(item_name, reason),
            ]))

            if target is not None:
                # transfer to another agent
                if type(target.object) == Agent:
                    target_inv = InventoryHelper(ctx, target)
                    ctx.dst_cmd(ctx.cond(ctx.metadata("item").eq(item_id), [
                        *target_inv.add(item_name, reason)
                    ]))
                # transfer to a converter
                if type(target.object) == Converter:
                    target.object.on_convert(ctx, item_name, cnd)

        ctx.require({"or": prereqs})
