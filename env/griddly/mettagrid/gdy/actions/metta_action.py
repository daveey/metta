from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyActionBehavior


class MettaActionBehavior(GriddlyActionBehavior):
    def __init__(self, actor, target, cfg: OmegaConf):
        super().__init__(actor, target)
        self.cfg = cfg
        self.cost = cfg.cost

    def commands(self, ctx: BehaviorContext):
        ctx.require([
            ctx.actor.frozen.eq(0),
            ctx.actor.energy.gte(self.cost)
        ])
        ctx.cmd([
            ctx.actor.energy.sub(self.cost)
        ])

