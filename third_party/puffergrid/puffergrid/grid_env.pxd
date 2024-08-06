from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.vector cimport vector
from puffergrid.action cimport ActionHandler
from puffergrid.event cimport EventManager
from puffergrid.stats_tracker cimport StatsTracker
from puffergrid.grid_object cimport GridObjectBase, GridObjectId, GridObject, GridLocation, Orientation, Layer
from puffergrid.grid cimport Grid
from puffergrid.event cimport EventManager
from puffergrid.observation_encoder cimport ObservationEncoder
cimport numpy as cnp

from libc.stdio cimport printf

cdef class GridEnv:
    cdef:
        Grid *_grid
        EventManager _event_manager
        unsigned int _current_timestep

        list[ActionHandler] _action_handlers
        ObservationEncoder _obs_encoder

        unsigned short obs_width
        unsigned short obs_height

        vector[GridObjectBase*] _agents

        cnp.ndarray _observations_np
        int[:,:,:,:] _observations
        cnp.ndarray _dones_np
        char[:] _dones
        cnp.ndarray _rewards_np
        float[:] _rewards

        StatsTracker _stats

        list[string] _grid_features

    cdef void add_agent(self, GridObjectBase* agent)

    cdef void _compute_observations(self)
    cdef void _step(self, unsigned int[:,:] actions)

    cdef void _compute_observation(
        self,
        unsigned int observer_r,
        unsigned int observer_c,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation)

    cpdef void set_buffers(
        self,
        cnp.ndarray[int, ndim=4] observations,
        cnp.ndarray[char, ndim=1] dones,
        cnp.ndarray[float, ndim=1] rewards)


    cpdef grid(self)
    cpdef unsigned int current_timestep(self)
    cpdef unsigned int map_width(self)
    cpdef unsigned int map_height(self)
    cpdef list[str] grid_features(self)
    cpdef unsigned int num_actions(self)

    cpdef void reset(self)

    cpdef void step(self, unsigned int[:,:] actions)

    cpdef observe(
        self,
        GridObjectId observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation)

    cpdef observe_at(
        self,
        unsigned short row,
        unsigned short col,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation)
