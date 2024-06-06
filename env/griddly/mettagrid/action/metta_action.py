from typing import Callable
from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyActionBehavior
from env.griddly.mettagrid.util.energy_helper import EnergyHelper


class MettaActionBehavior(GriddlyActionBehavior):
    def __init__(
            self,
            actor, target, cfg: OmegaConf,
            callback:Callable[[BehaviorContext], None] = None):

        super().__init__(actor, target)
        self.cfg = cfg
        self.callback = callback

    def cost (self):
        return self.cfg.cost

    def commands(self, ctx: BehaviorContext, *args, **kwargs):
        energy = EnergyHelper(ctx, ctx.actor)
        ctx.require([
            ctx.actor.frozen.eq(0),
            *energy.has_energy(self.cost()),
        ])
        ctx.cmd([
            ctx.global_var(f"stats:action:{ctx.action.name}").incr(),
            *energy.use(self.cost(), ctx.action.name)
        ])
        if self.callback is not None:
            self.callback(ctx, *args, **kwargs)

