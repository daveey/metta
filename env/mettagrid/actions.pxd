
from libc.stdio cimport printf
from libcpp.string cimport string

from puffergrid.action cimport ActionHandler
from puffergrid.grid_object cimport GridObjectBase, GridLocation, GridObjectId, Orientation
from env.mettagrid.objects cimport ObjectType, Agent, Wall, Tree, GridLayer_Agent, GridLayer_Object
from puffergrid.action cimport ActionHandler, ActionArg

cdef class MettaActionHandler(ActionHandler):
    cdef string name

cdef class Move(MettaActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg)

cdef class Rotate(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg)

cdef class Eat(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg)
