# grid.pxd
from env.puffergrid.grid_object cimport GridObject
cimport numpy as cnp


cdef enum GridLayers:
    LAYER_AGENT = 0
    LAYER_OBJECT = 1
    LAYER_COUNT = 2

cdef struct Observer:
    unsigned int id
    unsigned int object_id
    unsigned int r
    unsigned int c

cdef class PufferGrid:
    cdef:
        int _map_width
        int _map_height
        int _num_agents
        int _max_timesteps
        int _obs_width
        int _obs_height
        int _num_features
        int _current_timestep

        unsigned int[:, :, :] _grid # layer, width, height
        unsigned int[:, :, :, :] _observations # agent, property, width, height
        unsigned char[:, :] _actions # agent, action_and_args
        float[:] _rewards
        char[:] _terminals
        char[:] _truncations
        char[:] _masks

        unsigned int[:] _agent_ids

        Observer[:] _observers
        unsigned int _num_observers
        unsigned int _observer_type_id


        list _objects
        dict _object_classes
        int _obserer_type_id

    cdef int _allocate_object_id(self)
    cdef void _add_observer(self, unsigned int id, unsigned int r, unsigned int c)

    cdef void _compute_observations(self)
    cdef void _compute_observation(self, Observer observer)

