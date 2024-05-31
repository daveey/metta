from types import SimpleNamespace
from typing import Callable, Dict

from env.griddly.builder.variable import GriddlyVariable

class BehaviorContext():
    def __init__(self, game, action, actor_obj: "GriddlyObject", target_obj: "GriddlyObject"):
        self.game = game
        self.action = action
        self.actor_obj = actor_obj
        self.target_obj = target_obj

        self._preconditions = []
        self._src_commands = []
        self._dst_commands = []

        def make_ns(obj, prefix):
            return SimpleNamespace(
                object=obj,
                **{
                    name: GriddlyVariable(f"{prefix}.{obj.name}:{name}")
                    for name in obj.properties.keys()
                },
                **{
                    name: lambda delay=0: {
                        "exec": {
                            **action,
                            "Delay": delay
                        }
                        } for name, action in obj._internal_actions.items()
                }
            )
        self.actor = make_ns(actor_obj, "src")
        self.target = make_ns(target_obj, "dst")

    def cond(self, cond, true_commands, false_commands):
        return {
            "if": {
                "Conditions": cond,
                "OnTrue": true_commands,
                "OnFalse": false_commands
            }
        }

    def cmd(self, cmd):
        if not isinstance(cmd, list):
            cmd = [cmd]
        self._src_commands.extend(cmd)

    def dst_cmd(self, cmd):
        if not isinstance(cmd, list):
            cmd = [cmd]
        self._dst_commands.extend(cmd)

    def require(self, req):
        if not isinstance(req, list):
            req = [req]
        self._preconditions.extend(req)

    def global_var(self, name) -> GriddlyVariable:
        if name not in self.game._global_vars:
            self.game._global_vars[name] = {
                "Name": name,
                "InitialValue": 0
            }
        return GriddlyVariable(name, self._src_commands)

    def player_var(self, name) -> GriddlyVariable:
        if name not in self.game._player_vars:
            self.game._player_vars[name] = {
                "Name": name,
                "InitialValue": 0,
                "PerPlayer": True
            }
        return GriddlyVariable(name, self._src_commands)


class GriddlyActionBehavior():
    def __init__(self, actor_object_id: str, target_object_id: str):
        # set in Action.register_behaviour
        self.action = None
        self.game = None

        self.actor_object_id = actor_object_id
        self.target_object_id = target_object_id

    def commands(self, ctx: BehaviorContext):
        return []

    def build(self):
        ctx = BehaviorContext(
            self.game,
            self.action,
            self.game.object(self.actor_object_id),
            self.game.object(self.target_object_id)
        )
        self.commands(ctx)

        src = {
            "Object": self.actor_object_id,
        }

        src["Preconditions"] = ctx._preconditions
        src["Commands"] = ctx._src_commands

        dst = {
            "Object": [self.target_object_id],
        }
        dst["Commands"] = ctx._dst_commands

        return {
            "Src": src,
            "Dst": dst
        }

class GriddlyActionInput():
    def __init__(self, dest=None, rot=None, description="", metadata = {}):
        self.dest = dest
        self.rot = rot
        self.description = description
        self.metadata = metadata

    def build(self):
        inputs = {
            "Description": self.description,
        }
        if self.metadata:
            inputs["Metadata"] = self.metadata
        if self.dest:
            inputs["VectorToDest"] = self.dest
        if self.rot:
            inputs["OrientationVector"] = self.rot

        return inputs

class GriddlyAction():
    def __init__(
            self,
            name: str,
            game,
            inputs: Dict[str, GriddlyActionInput],
            relative = True,
            internal = False
        ):

        self.name = name
        self.game = game
        self.inputs = inputs
        self.behaviours = []
        self.relative = relative
        self.internal = internal

    def add_behaviour(self, behaviour):
        self.behaviours.append(behaviour)
        behaviour.action = self
        behaviour.game = self.game

    def build(self):
        action = {
            "Name": self.name,
            "InputMapping": {
                "Inputs": { id + 1: input.build() for id, input in enumerate(self.inputs) },
                "Relative": self.relative,
                "Internal": True if self.internal else False,
            },
            "Behaviours": [ b.build() for b in self.behaviours ],
        }
        return action

    def src(self, name):
        return f"{self.name}:{name}"

class GriddlyInternalAction(GriddlyAction):
    class Behavior(GriddlyActionBehavior):
        def __init__(self, actor, target, callback: Callable[[BehaviorContext], None]):
            super().__init__(actor, target)
            self.commands = callback

        def commands(self, ctx: BehaviorContext):
            ctx.cmd(self.callback(ctx))

    def __init__(
            self,
            game,
            object_id: str,
            name: str,
            callback: Callable[[BehaviorContext], None],
            choices = None,
        ):

        if choices is None:
            choices = [1]

        super().__init__(
            name=f"{object_id}:{name}",
            game=game,
            inputs=[
                GriddlyActionInput(dest=[0, 0], metadata={"choice": c}) for c in choices
            ],
            internal = True,
        )
        self.add_behaviour(
            GriddlyInternalAction.Behavior(object_id, object_id, callback))

