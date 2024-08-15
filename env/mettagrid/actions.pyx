
from libc.stdio cimport printf

from puffergrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from env.mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class MettaActionHandler(ActionHandler):
    def __init__(self, action_name, action_cost=0):
        self.action_name = action_name

        self._stats.action = "action." + action_name
        self._stats.action_energy = "action." + action_name + ".energy"

        for t, n in enumerate(ObjectTypeNames):
            self._stats.target[t] = self._stats.action + "." + n
            self._stats.target_energy[t] = self._stats.action_energy + "." + n

        self.action_cost = action_cost

    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef Agent *actor = <Agent*>self.env._grid.object(actor_object_id)
        if actor.energy < self.action_cost:
            return False

        actor.energy -= self.action_cost
        self.env._stats.agent_add(actor_id, self._stats.action_energy.c_str(), self.action_cost)

        cdef char result = self._handle_action(actor_id, actor, arg)

        if result:
            self.env._stats.agent_incr(actor_id, self._stats.action.c_str())

        return result

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False

cdef class Move(MettaActionHandler):
    def __init__(self):
        MettaActionHandler.__init__(self, "move")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef unsigned short direction = arg
        if direction >= 2:
            return False

        cdef Orientation orientation = <Orientation>((actor.orientation + 2*(direction)) % 4)
        cdef GridLocation old_loc = actor.location
        cdef GridLocation new_loc = self.env._grid.relative_location(old_loc, orientation)
        if not self.env._grid.is_empty(new_loc.r, new_loc.c):
            return False
        return self.env._grid.move_object(actor.id, new_loc)

cdef class Rotate(MettaActionHandler):
    def __init__(self):
        MettaActionHandler.__init__(self, "rotate")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef unsigned short orientation = arg
        if orientation >= 4:
            return False

        actor.orientation = orientation
        return True

cdef class Use(MettaActionHandler):
    def __init__(self):
        MettaActionHandler.__init__(self, "use")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        cdef GridLocation target_loc = self.env._grid.relative_location(
            actor.location,
            <Orientation>actor.orientation
        )
        target_loc.layer = GridLayer.Object_Layer
        cdef MettaObject *target = <MettaObject*>self.env._grid.object_at(target_loc)
        if target == NULL:
            return False

        if not target.usable(actor):
            return False

        cdef Usable *usable = <Usable*> target
        cdef Generator *generator
        cdef Converter *converter

        actor.energy -= usable.energy_cost
        usable.ready = 0

        self.env._stats.agent_incr(actor_id, self._stats.target[target._type_id].c_str())
        self.env._stats.agent_add(actor_id, self._stats.target_energy[target._type_id].c_str(), usable.energy_cost + self.action_cost)

        if target._type_id == ObjectType.AltarT:
            self.env._rewards[actor_id] += 10
            self.env._stats.agent_add(actor_id, "reward", 10)
            self.env._stats.game_add("reward", 10)

        if target._type_id == ObjectType.GeneratorT:
            generator = <Generator*>target
            generator.r1 -= 1
            actor.update_inventory(InventoryItem.r1, 1)
            self.env._stats.agent_incr(actor_id, "r1.gained")
            self.env._stats.game_incr("r1.harvested")

        if target._type_id == ObjectType.ConverterT:
            converter = <Converter*>target
            actor.update_inventory(converter.input_resource, -1)
            self.env._stats.agent_incr(actor_id, InventoryItemNames[converter.input_resource] + ".used")

            actor.update_inventory(converter.output_resource, 1)
            self.env._stats.agent_incr(actor_id, InventoryItemNames[converter.input_resource] + ".gained")

            actor.energy += converter.output_energy
            self.env._stats.agent_add(actor_id, "energy.gained", converter.output_energy)

        return True

cdef class Attack(MettaActionHandler):
    def __init__(self):
        MettaActionHandler.__init__(self, "attack")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False

cdef class ToggleShield(MettaActionHandler):
    def __init__(self):
        MettaActionHandler.__init__(self, "shield")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        if actor.shield:
            actor.shield = True
            actor.energy_upkeep += 1
        else:
            actor.shield = False
            actor.energy_upkeep -= 1

cdef class Gift(MettaActionHandler):
    def __init__(self):
        MettaActionHandler.__init__(self, "gift")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):
        return False



