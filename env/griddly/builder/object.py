

from typing import Callable, List

from env.griddly.builder.action import BehaviorContext, GriddlyInternalAction

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
        self._internal_actions = {}

    def build(self):
        object_spec = {
            "Name": self.name,
            "Z": self.z,
            "MapCharacter": self.symbol,
            "Variables": self._variables(),
            "Observers": self._observers(),
            "InitialActions": [self._internal_actions[a] for a in self._initial_actions]
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
        self._internal_actions[name] = {
            "Action": f"{self.name}:{name}",
            "Randomize": True if (choices and len(choices) > 0) else False,
            "ActionId": 1
        }

        if initial:
            self._initial_actions.append(name)

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
