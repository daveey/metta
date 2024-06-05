

from sympy import N
from env.griddly.builder.action import BehaviorContext

class InventoryHelper():

    Resources = ["r1", "r2", "r3"]

    Items = {
        id: item for id, item in enumerate(Resources)
    }

    Properties = {
        f"inv_{item}": 0 for item in Items.values()
    }

    def __init__(self, ctx: BehaviorContext, agent) -> None:
        self.agent = agent
        self.ctx = ctx
        self.max_inventory = agent.object.cfg.max_inventory

    def item_var(self, item: str):
        return self.agent.__dict__[f"inv_{item}"]

    def has_space(self, item: str):
        return [ self.item_var(item).lt(self.max_inventory) ]

    def has_item(self, item: str, amount: str = None):
        amount = amount or 1
        return self.item_var(item).gte(amount)

    def add(self, item: str, reason: str, amount: str = None):
        amount = amount or 1

        return [
                self.item_var(item).add(amount),
                self.ctx.player_var(f"stats:agent:inv:{item}:gained").add(amount),
                self.ctx.player_var(f"stats:agent:inv:{item}:gained:{reason}").add(amount),
                self.ctx.cond(self.item_var(item).gt(self.max_inventory), [
                    self.ctx.player_var(f"stats:agent:inv:{item}:lost:full").add(self.item_var(item).val()),
                    self.ctx.player_var(f"stats:agent:inv:{item}:lost:full").sub(self.max_inventory),
                    self.item_var(item).set(self.max_inventory)
                ])
            ]

    def remove(self, item: str, reason: str, amount: str = None):
        amount = amount or 1

        return [ self.ctx.cond(
                self.has_item(item, amount), [
                    self.item_var(item).sub(amount),
                    self.ctx.player_var(f"stats:agent:inv:{item}:lost:{reason}").add(amount),
                ])
        ]
