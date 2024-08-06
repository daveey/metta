from puffergrid.grid_object cimport GridObjectId
from puffergrid.grid_env cimport GridEnv

ctypedef unsigned int ActionArg

cdef class ActionHandler:
    cdef GridEnv env

    cdef void init(self, GridEnv env)

    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg)

