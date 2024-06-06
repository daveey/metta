from typing import Dict, List
import numpy as np
import yaml
from env.griddly.builder.action import GriddlyAction
from env.griddly.builder.object import GriddlyObject

class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True

class GriddlyGameBuilder():
    def __init__(
            self,
            obs_width: int,
            obs_height: int,
            tile_size: int,
            width: int,
            height: int,
            num_agents: int,
            max_steps: int):
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.tile_size = tile_size
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.max_steps = max_steps
        self._objects ={}
        self._actions = {}
        self._global_vars = {}

        self.register_object(
            GriddlyObject(self, "_empty", " ", sprites=["oryx/oryx_fantasy/floor1-0.png"]))

    def build(self):
        objects = [obj.build() for obj in self._objects.values() if obj.name != "_empty"]
        actions = [action.build() for action in self._actions.values()]

        griddly_cfg = {
            "Version": "0.1",
            "Environment": {
                "Name": "MettaGrid",
                "Description": "MettaGrid",
                "Player": {
                    "Count": self.num_agents,
                    "AvatarObject": "agent"
                },
                "Observers": {
                    "GlobalSpriteObserver": {
                        "TileSize": self.tile_size,
                        "Type": "SPRITE_2D",
                        "RotateAvatarImage": True,
                        "BackgroundTile": "oryx/oryx_fantasy/floor1-0.png",
                        "Shader": {
                            "ObjectVariables": [
                                "agent:energy",
                                "agent:energy"
                            ],
                            "ObserverAvatarMode": "DARKEN",
                        }
                    },
                    "VectorAgent": {
                        "Type": "VECTOR",
                        "Width": self.obs_width,
                        "Height": self.obs_height,
                        "TrackAvatar": True,
                        "RotateWithAvatar": True,
                        "IncludeVariables": True
                    }
                },
                "Variables": list(self._global_vars.values()),
                "Levels": ["\n".join(["  ".join(row) for row in self.level()])],
                "Termination": []
            },
            "Objects": objects,
            "Actions": actions,
        }
        return yaml.dump(
            griddly_cfg,
            Dumper=NoAliasDumper,
            sort_keys=False,
            default_flow_style=False
        )

    def register_object(self, obj: GriddlyObject):
        self._objects[obj.name] = obj

    def register_action(self, action: GriddlyAction):
        self._actions[action.name] = action

    def level(self) -> np.ndarray:
        return np.array([["."] * self.width] * self.height).astype("U6")

    def objects(self) -> List[GriddlyObject]:
        return [
            o for o in self._objects.values() if o.name != "_empty"
        ]

    def object(self, name: str) -> GriddlyObject:
        return self._objects[name]

    def actions(self) -> List[GriddlyAction]:
        return list(self._actions.values())

    def register_global_variable(self, name: str, per_player: bool = False):
        if name not in self._global_vars:
            self._global_vars[name] = {
                "Name": name,
                "InitialValue": 0,
                "PerPlayer": per_player
            }
