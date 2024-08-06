from puffergrid.grid_object cimport GridObjectId
from puffergrid.grid_env cimport GridEnv
from puffergrid.action cimport ActionArg

cdef class ActionHandler:

    cdef void init(self, GridEnv env):
        self.env = env

    cdef char handle_action(
        self,
        unsigned int actor_id,
        GridObjectId actor_object_id,
        ActionArg arg):
        return False

