
from libc.stdio cimport printf

from puffergrid.action cimport ActionHandler
from puffergrid.grid_object cimport GridObjectId, TypeId
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport Agent
from libcpp.string cimport string
from libcpp.map cimport map

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

cdef class Move(MettaActionHandler):
    pass

cdef class Rotate(MettaActionHandler):
    pass
cdef class Use(MettaActionHandler):
    pass
cdef class Attack(MettaActionHandler):
    pass
cdef class ToggleShield(MettaActionHandler):
    pass
cdef class Gift(MettaActionHandler):
    pass
