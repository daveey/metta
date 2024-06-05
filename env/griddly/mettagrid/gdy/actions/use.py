from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyAction, GriddlyActionInput
from env.griddly.builder.game import GriddlyGame
from env.griddly.builder.variable import GriddlyVariable
from env.griddly.mettagrid.gdy.actions.metta_action import MettaActionBehavior


class Use(GriddlyAction):
    def __init__(
            self,
            game: GriddlyGame,
            cfg: OmegaConf):

        super().__init__(
            name="use",
            game=game,
            inputs=[
                GriddlyActionInput(dest=[0, -1], description="Use Object"),
            ],
        )
        self.cfg = cfg

        self.add_behaviour(UseBehavior(game, "agent", "generator", cfg))
        self.add_behaviour(UseBehavior(game, "agent", "altar", cfg))
        self.add_behaviour(UseBehavior(game, "agent", "converter", cfg))


class UseBehavior(MettaActionBehavior):
    def __init__(self, game, actor, target, cfg):
        self.cfg = game.action_configs.use
        super().__init__(actor, target, cfg)
        self.cost += game.object_configs[target].cost

    def commands(self, ctx: BehaviorContext):
        super().commands(ctx)
        ctx.require(ctx.target.state.eq(ctx.target.object.States.ready))
        ctx.cmd(ctx.global_var(f"stats:{ctx.target.object.name}:used").incr())
        ctx.target.object.on_use(ctx)
