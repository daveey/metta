
from libc.stdio cimport printf

from puffergrid.action cimport ActionHandler
from puffergrid.grid_object cimport GridObjectBase, GridLocation, GridObjectId, Orientation
from env.mettagrid.objects cimport ObjectType, Agent, Wall, Tree, GridLayer_Agent, GridLayer_Object
from puffergrid.action cimport ActionHandler, ActionArg

cdef class MoveHandler(ActionHandler):
    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg)