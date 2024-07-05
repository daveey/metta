# grid.pxd
from env.puffergrid.object cimport GridObject


cdef enum GridLayers:
    LAYER_AGENT = 0
    LAYER_OBJECT = 1
    LAYER_COUNT = 2

cdef class PufferGrid:
    cdef:
        int map_width
        int map_height
        int num_agents
        int max_timesteps
        int obs_width
        int obs_height
        int num_features
        int current_timestep

        unsigned int[:, :, :] grid # layer, width, height
        unsigned int[:, :, :, :] observations # agent, property, width, height
        float[:] rewards
        char[:] terminals
        char[:] truncations
        char[:] masks

        list agents
        list objects
        int next_object_id

    cdef void _compute_observations(self)
    cdef int _allocate_object_id(self)
