from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyAction, GriddlyActionInput
from env.griddly.builder.game_builder import GriddlyGameBuilder
from env.griddly.builder.variable import GriddlyVariable
from env.griddly.mettagrid.action.metta_action import MettaActionBehavior
from env.griddly.mettagrid.util.energy_helper import EnergyHelper


class Use(GriddlyAction):
    def __init__(
            self,
            game: GriddlyGameBuilder,
            cfg: OmegaConf):

        super().__init__(
            name="use",
            game=game,
            inputs=[
                GriddlyActionInput(dest=[0, -1], description="Use Object"),
            ],
        )
        self.cfg = cfg

        self.add_behaviour(MettaActionBehavior("agent", "generator", cfg, self.use))
        self.add_behaviour(MettaActionBehavior("agent", "altar", cfg, self.use))
        self.add_behaviour(MettaActionBehavior("agent", "converter", cfg, self.use))
        self.add_behaviour(MettaActionBehavior("agent", "agent", cfg, self.use))

    def use(self, ctx: BehaviorContext):
        energy_helper = EnergyHelper(ctx, ctx.actor)
        ctx.require([
            energy_helper.has_energy(self.cfg.cost + ctx.target.object.cfg.use_cost),
        ])
        ctx.require(ctx.target.object.usable(ctx))

        ctx.cmd([
            ctx.global_var(f"stats:{ctx.target.object.name}:used").incr(),
            energy_helper.use(self.cfg.cost + ctx.target.object.cfg.use_cost, f"{ctx.target.object.name}")
        ])
        ctx.target.object.on_use(ctx)

