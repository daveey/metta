
from libc.stdio cimport printf

from puffergrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from env.mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from env.mettagrid.actions.actions cimport MettaActionHandler

cdef class Attack(MettaActionHandler):
    def __init__(self):
        MettaActionHandler.__init__(self, "attack")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef short attack_damage = 10

        if arg > 9 or arg < 1:
            return False

        cdef short distance = 0
        cdef short offset = 0
        distance = 1 + (arg - 1) // 3
        offset = (arg - 1) % 3 - 1

        cdef GridLocation target_loc = self.env._grid.relative_location(
            actor.location,
            <Orientation>actor.orientation,
            distance, offset)

        target_loc.layer = GridLayer.Agent_Layer
        cdef Agent * agent_target = <Agent *>self.env._grid.object_at(target_loc)
        if agent_target:
            self.env._stats.agent_incr(actor_id, self._stats.target[agent_target._type_id].c_str())
            if agent_target.shield and agent_target.energy >= attack_damage:
                    agent_target.energy -= attack_damage
                    self.env._stats.agent_add(actor_id, "shield_damage", attack_damage)
            else:
                self.env._stats.agent_add(actor_id, "shield_damage", agent_target.energy)
                agent_target.energy = 0
                agent_target.shield = False
                agent_target.frozen = True
                self.env._stats.agent_incr(actor_id, "attack.frozen")
            return True

        target_loc.layer = GridLayer.Object_Layer
        cdef MettaObject * object_target = <MettaObject *>self.env._grid.object_at(target_loc)
        if object_target:
            self.env._stats.agent_incr(actor_id, self._stats.target[object_target._type_id].c_str())
            object_target.hp -= 1
            self.env._stats.agent_incr(actor_id, "damage." + ObjectTypeNames[object_target._type_id])
            if object_target.hp <= 0:
                self.env._grid.remove_object(object_target)
                self.env._stats.agent_incr(actor_id, "destroyed." + ObjectTypeNames[object_target._type_id])

            return True

        return False
