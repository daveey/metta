
from libc.stdio cimport printf

from puffergrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport MettaObject, Usable, Altar, Attackable, Agent, Events, GridLayer

cdef class Move(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef unsigned short direction = arg
        if direction >= 2:
            return False

        cdef Agent* agent = <Agent*>self.env._grid.object(actor_object_id)
        cdef Orientation orientation = <Orientation>((agent.orientation + 2*(direction)) % 4)
        cdef GridLocation old_loc = agent.location
        cdef GridLocation new_loc = self.env._grid.relative_location(old_loc, orientation)
        if not self.env._grid.is_empty(new_loc.r, new_loc.c):
            return False
        self.env._grid.move_object(actor_object_id, new_loc)
        if actor_id == 0:
            print("move success")

        return True

cdef class Rotate(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef unsigned short orientation = arg
        if orientation >= 4:
            return False

        cdef Agent* agent = <Agent*>self.env._grid.object(actor_object_id)
        agent.orientation = orientation
        if actor_id == 0:
            print("rotating", orientation)
        #self.env._stats.agent_incr(actor_id, "action.rotate")
        return True

cdef class Use(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef Agent* agent = <Agent*>self.env._grid.object(actor_object_id)
        cdef GridLocation target_loc = self.env._grid.relative_location(
            agent.location,
            <Orientation>agent.orientation
        )
        target_loc.layer = GridLayer.Object_Layer
        cdef GridObject *target = self.env._grid.object_at(target_loc)
        cdef Usable *usable
        if target != NULL:
            if (<MettaObject*>target).usable():
                printf("object usable")
                #usable = <Usable*>target
                #usable.on_use()
            self.env._stats.agent_incr(actor_id, "action.use")
        elif target == NULL:
            if actor_id == 0:
                print("failed null")
            pass
            #self.env._stats.agent_incr(actor_id, "action.use.null")
        else:
            self.env._stats.agent_incr(actor_id, "action.use." + target._type_id)

        return False

cdef class Attack(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):
        return False

cdef class ToggleShield(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):
        return False

cdef class Gift(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):
        return False



