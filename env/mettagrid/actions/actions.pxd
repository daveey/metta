
from libc.stdio cimport printf

from puffergrid.action cimport ActionHandler
from puffergrid.grid_object cimport GridObjectId, TypeId
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport Agent
from libcpp.string cimport string
from libcpp.map cimport map
from puffergrid.grid_object cimport GridLocation, GridObjectId, GridObject
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from env.mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames

cdef struct StatNames:
    string action
    string action_energy
    map[TypeId, string] target
    map[TypeId, string] target_energy

cdef class MettaActionHandler(ActionHandler):
    cdef StatNames _stats
    cdef string action_name
    cdef int action_cost


    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg)

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg)
