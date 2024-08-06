
from libc.stdio cimport printf

from puffergrid.grid_object cimport GridLocation, GridObjectId, Orientation
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport ObjectType, Agent, ResetTree, Tree

cdef class Move(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef unsigned short direction = arg
        if direction >= 2:
            return False

        cdef Agent* agent = self.env._grid.object[Agent](actor_object_id)
        cdef Orientation orientation = <Orientation>((agent.props.orientation + 2*(direction)) % 4)
        cdef GridLocation old_loc = agent.location
        cdef GridLocation new_loc = self.env._grid.relative_location(old_loc, orientation)
        if not self.env._grid.is_empty(new_loc.r, new_loc.c):
            return False
        cdef char s = self.env._grid.move_object(actor_object_id, new_loc)
        return s

cdef class Rotate(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):

        cdef unsigned short orientation = arg
        if orientation >= 4:
            return False

        cdef Agent* agent = self.env._grid.object[Agent](actor_object_id)
        agent.props.orientation = orientation
        return True

cdef class Eat(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):
        cdef Tree *tree = NULL
        cdef Agent* agent = self.env._grid.object[Agent](actor_object_id)
        cdef GridLocation target_loc = self.env._grid.relative_location(
            agent.location,
            <Orientation>agent.props.orientation
        )
        tree = self.env._grid.object_at[Tree](target_loc.r, target_loc.c, ObjectType.TreeT)
        if tree == NULL or tree.props.has_fruit == 0:
            return False

        tree.props.has_fruit = 0
        agent.props.energy += 10
        self.env._rewards[actor_id] += 10
        # printf("Agent %d ate a fruit\n", actor_id)
        self.env._event_manager.schedule_event(100, 0, tree.id, 0)
        return True
