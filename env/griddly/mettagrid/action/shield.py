from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyAction, GriddlyActionInput
from env.griddly.builder.game_builder import GriddlyGameBuilder
from env.griddly.mettagrid.action.metta_action import MettaActionBehavior

class Shield(GriddlyAction):
    def __init__(
            self,
            game: GriddlyGameBuilder,
            cfg: OmegaConf):

        super().__init__(
            name="shield",
            game=game,
            inputs=[
                GriddlyActionInput(dest=[0, 0], description="Toggle Shield"),
            ],
            relative=False
        )

        self.add_behaviour(MettaActionBehavior("agent", "agent", cfg, self.shield))

    def shield(self, ctx: BehaviorContext):
        ctx.cmd(
            ctx.cond(
                ctx.actor.shield.eq(0),
                ctx.actor.object.shield_on_cmds(ctx),
                ctx.actor.object.shield_off_cmds(ctx)
        ))
