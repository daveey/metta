# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
# distutils: language = c++
from libcpp.vector cimport vector
cimport numpy as cnp
from env.puffergrid.grid_object cimport GridObject, GridCoordinates, GridLocation
from libc.stdlib cimport malloc, free
from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair
from libcpp cimport bool

cdef extern from "grid.h":
    cdef struct Event:
        unsigned int timestamp
        unsigned short event_id
        unsigned short object_id
        unsigned int arg

cdef struct TypeInfo:
    unsigned int type_id
    unsigned short grid_layer
    unsigned int object_size
    unsigned int num_properties
    unsigned int obs_offset

cdef class PufferGrid:
    cdef:
        unsigned int _map_width
        unsigned int _map_height
        unsigned int _num_layers
        unsigned int _max_timesteps

        cnp.dtype _grid_dtype

        unsigned int _num_features
        unsigned int _current_timestep

        # cnp.ndarray _grid_data
        # vector[vector[vector[unsigned int]]] _grid # width, height, layer
        cnp.ndarray _np_grid
        unsigned int[:, :, :] _grid

        dict _type_ids
        dict _object_dtypes
        list _object_data
        vector[GridObject] _objects
        vector[TypeInfo] _object_types

        dict _object_classes
        int _obserer_type_id

        priority_queue[Event] _event_queue

    cdef unsigned int _add_object(
        self,
        unsigned int object_id,
        TypeInfo type_info,
        unsigned int r, unsigned int c,
        cnp.ndarray data)

    cdef void _compute_observation(
        self,
        unsigned int observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        unsigned int[:,:,:] observation)

    cdef void _process_events(self)

    cpdef get_grid(self)
    cpdef unsigned int get_current_timestep(self)
    cpdef unsigned int get_num_features(self)
    cpdef void move_object(self, unsigned int obj_id, unsigned int new_r, unsigned int new_c)
    cdef GridObject _get_object(self, unsigned int obj_id)
    cpdef GridLocation location(self, int object_id)

    cdef GridLocation _relative_location(self, GridLocation loc, unsigned short orientation)
    cpdef void compute_observations(self,
        unsigned int[:] observer_ids,
        unsigned short obs_width,
        unsigned short obs_height,
        unsigned int[:,:,:,:] obs)

    cpdef void step(
        self,
        unsigned int[:] actor_ids,
        unsigned int[:,:] actions,
        float[:] rewards,
        char[:] dones)

    cdef void handle_action(
        self,
        unsigned int actor_id,
        unsigned short action_id,
        unsigned short action_arg,
        float *reward,
        char *done)

    cdef void _schedule_event(
        self,
        unsigned int delay,
        unsigned short event_id,
        unsigned short object_id,
        int arg)

    cdef void handle_event(self, Event event)

