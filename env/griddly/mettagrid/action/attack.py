from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyAction, GriddlyActionInput
from env.griddly.builder.game_builder import GriddlyGameBuilder
from env.griddly.builder.variable import GriddlyVariable
from env.griddly.mettagrid.action.metta_action import MettaActionBehavior


class Attack(GriddlyAction):
    def __init__(
            self,
            game: GriddlyGameBuilder,
            cfg: OmegaConf):

        inputs = [GriddlyActionInput(dest=[dx, dy], description=f"Attack {i+1}")
            for i, (dx, dy) in enumerate((dx, dy)
                for dy in [-1, -2, -3] for dx in [-1, 0, 1])]

        super().__init__(
            name="attack",
            game=game,
            inputs=inputs,
        )
        self.cfg = cfg

        for target in ["generator", "altar", "converter", "agent", "wall"]:
            self.add_behaviour(MettaActionBehavior("agent", target, cfg, self.attack))

    def attack(self, ctx: BehaviorContext):
        ctx.cmd(ctx.global_var(f"stats:{ctx.target.object.name}:attacked").incr())
        ctx.target.object.on_attack(ctx, self.cfg.damage)
