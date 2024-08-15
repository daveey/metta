
from libc.stdio cimport printf

from omegaconf import OmegaConf

from puffergrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from env.mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from env.mettagrid.actions.actions cimport MettaActionHandler

cdef class Use(MettaActionHandler):
    def __init__(self, cfg: OmegaConf):
        MettaActionHandler.__init__(self, cfg, "use")

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
        actor.energy -= usable.use_cost

        usable.ready = 0
        self.env._event_manager.schedule_event(Events.Reset, usable.cooldown, usable.id, 0)

        self.env._stats.agent_incr(actor_id, self._stats.target[target._type_id].c_str())
        self.env._stats.agent_add(actor_id, self._stats.target_energy[target._type_id].c_str(), usable.use_cost + self.action_cost)

        if target._type_id == ObjectType.AltarT:
            self.env._rewards[actor_id] += usable.use_cost
            self.env._stats.agent_add(actor_id, "reward", usable.use_cost)
            self.env._stats.game_add("reward", usable.use_cost)

        cdef Generator *generator
        if target._type_id == ObjectType.GeneratorT:
            generator = <Generator*>target
            generator.r1 -= 1
            actor.update_inventory(InventoryItem.r1, 1)
            self.env._stats.agent_incr(actor_id, "r1.gained")
            self.env._stats.game_incr("r1.harvested")

        cdef Converter *converter
        if target._type_id == ObjectType.ConverterT:
            converter = <Converter*>target
            actor.update_inventory(converter.input_resource, -1)
            self.env._stats.agent_incr(actor_id, InventoryItemNames[converter.input_resource] + ".used")

            actor.update_inventory(converter.output_resource, 1)
            self.env._stats.agent_incr(actor_id, InventoryItemNames[converter.input_resource] + ".gained")

            actor.energy += converter.output_energy
            self.env._stats.agent_add(actor_id, "energy.gained", converter.output_energy)

        return True
