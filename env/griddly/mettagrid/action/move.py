from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyAction, GriddlyActionBehavior, GriddlyActionInput
from env.griddly.builder.game_builder import GriddlyGameBuilder
from env.griddly.mettagrid.action.metta_action import MettaActionBehavior


class Move(GriddlyAction):
    def __init__(
            self,
            game: GriddlyGameBuilder,
            cfg: OmegaConf):

        self.cfg = cfg

        super().__init__(
            name="move",
            game=game,
            inputs=[
                GriddlyActionInput(dest=[0, -1], description="Move forward"),
                GriddlyActionInput(dest=[0, 1], description="Move backward"),
            ],
        )

        self.add_behaviour(MettaActionBehavior("agent", "_empty", cfg, self.move))

    def move(self, ctx: BehaviorContext):
        ctx.cmd({"mov": "_dest"})
