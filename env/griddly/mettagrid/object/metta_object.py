
from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext
from env.griddly.builder.object import GriddlyObject


class MettaObject(GriddlyObject):
    def __init__(self, game, cfg: OmegaConf, **kwargs):
        self.cfg = cfg
        properties = kwargs.get("properties", {})
        properties["hp"] = cfg.hp

        super().__init__(
            game=game,
            **kwargs
        )

    def on_attack(self, ctx: BehaviorContext, damage: int):
        # most objects take 1 damage per attack
        damage = 1

        ctx.dst_cmd([
            ctx.target.hp.sub(damage),
            ctx.cond(ctx.target.hp.eq(0), [
                ctx.global_var(f"stats:{ctx.target.object.name}:destroyed").incr(),
                {"remove": True }
            ])
        ])

    def usable(self, ctx: BehaviorContext):
        return [ctx.target.state.eq(ctx.target.object.States.ready)]
