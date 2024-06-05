

from types import SimpleNamespace
from typing import Callable, List

from matplotlib.pylab import f

from env.griddly.builder.action import BehaviorContext, GriddlyInternalAction
from env.griddly.builder.variable import GriddlyVariable

class GriddlyObject():
    def __init__(
            self,
            game,
            name,
            symbol,
            sprites=["oryx/oryx_fantasy/wall2-0.png"],
            tiling_mode=None,
            properties = {}
        ):

        self.game = game
        self.name = name
        self.sprites = sprites
        self.z = 0
        self.symbol = symbol
        self.tiling_mode = tiling_mode
        self.properties = properties
        self._initial_actions = []
        self._internal_actions = []

    def build(self):
        object_spec = {
            "Name": self.name,
            "Z": self.z,
            "MapCharacter": self.symbol,
            "Variables": self._variables(),
            "Observers": self._observers(),
            "InitialActions": [self.actions()[a](randomize=True)["exec"] for a in self._initial_actions]
        }
        return object_spec

    def register_action(self, name: str, callback: Callable[["BehaviorContext"], None], choices: List[int] = None, initial=False):
        self.game.register_action(GriddlyInternalAction(
            game=self.game,
            object_id=self.name,
            name=name,
            callback=callback,
            choices=choices
        ))
        self._internal_actions.append(name)
        if initial:
            self._initial_actions.append(name)

    def _make_action_call(self, name: str):
        return lambda delay=0, input=1, randomize=False : {
            "exec": {
                "Action": f"{self.name}:{name}",
                "ActionId": input,
                "Delay": delay,
                "Randomize": randomize
            }
        }

    def actions(self):
        return {
            name: self._make_action_call(name)
            for name in self._internal_actions
        }

    def variables(self, prefix: str):
        return {
            name: GriddlyVariable(f"{prefix}.{self.name}:{name}")
            for name in self.properties.keys()
        }

    def _action_namespace(self, prefix: str):
        return SimpleNamespace(
            object=self,
            **self.variables(prefix),
            **self.actions(),
        )

    def action_actor(self):
        return self._action_namespace("src")

    def action_target(self):
        return self._action_namespace("dst")

    def _variables(self):
        return [{
            "Name": f"{self.name}:{k}",
            "InitialValue": v
        } for k,v in self.properties.items()]

    def _observers(self):
        if self.tiling_mode == "WALL_16":
            return {
                "GlobalSpriteObserver": [{
                    "TilingMode": "WALL_16",
                    "Image": self.sprites,
                }]
            }
        return {
            "GlobalSpriteObserver": [{"Image": s} for s in self.sprites]
        }
