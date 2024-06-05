from omegaconf import OmegaConf
from env.griddly.builder.action import BehaviorContext, GriddlyAction, GriddlyActionInput
from env.griddly.builder.game import GriddlyGame
from env.griddly.mettagrid.gdy.actions.metta_action import MettaActionBehavior

class Rotate(GriddlyAction):
    def __init__(
            self,
            game: GriddlyGame,
            cfg: OmegaConf):

        super().__init__(
            name="rotate",
            game=game,
            inputs=[
                GriddlyActionInput(rot=[0, 1], metadata={"dir": 1}, description="Rotate down"),
                GriddlyActionInput(rot=[-1, 0], metadata={"dir": 2}, description="Rotate left"),
                GriddlyActionInput(rot=[1, 0], metadata={"dir": 3}, description="Rotate right"),
                GriddlyActionInput(rot=[0, -1], metadata={"dir": 4}, description="Rotate up"),
            ],
            relative=False
        )

        self.add_behaviour(MettaActionBehavior("agent", "agent", cfg, self.rotate))

    def rotate(self, ctx: BehaviorContext):
        ctx.cmd([
            {"rot": "_dir"},
            ctx.actor.dir.set("meta.dir")
        ])
