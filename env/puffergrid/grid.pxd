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
from env.puffergrid.grid_object cimport GridObject, GridLocation
from libcpp.queue cimport priority_queue
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "grid.h":
    cdef struct Event:
        unsigned int timestamp
        unsigned short event_id
        unsigned short object_id
        unsigned int arg

cdef struct Action:
        unsigned short id
        unsigned int actor_id
        unsigned int agent_idx
        unsigned short arg

cdef struct TypeInfo:
    unsigned int type_id
    unsigned int object_size
    unsigned int num_properties
    unsigned int obs_offset

cdef class PufferGrid:
    cdef:
        unsigned int _map_width
        unsigned int _map_height
        unsigned short _num_layers

        unsigned int _num_features
        unsigned int _current_timestep

        cnp.ndarray _np_grid
        int[:, :, :] _grid
        cnp.ndarray _fake_props
        const int[:] _fake_props_view

        dict _type_ids
        dict _object_dtypes
        list _object_data
        vector[GridObject] _objects
        vector[TypeInfo] _object_types
        list[string] _grid_features

        dict _object_classes
        int _obserer_type_id

        priority_queue[Event] _event_queue

    cdef unsigned int _add_object(
        self,
        unsigned int object_id,
        unsigned int type_id,
        GridLocation location,
        cnp.ndarray data)

    cdef void _compute_observation(
        self,
        unsigned int observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation)

    cdef void _process_events(self)
    cdef GridObject * _get_object(self, unsigned int obj_id)

    cpdef grid(self)
    cpdef unsigned int current_timestep(self)
    cpdef unsigned int num_features(self)
    cpdef unsigned int map_width(self)
    cpdef unsigned int map_height(self)
    cpdef list[str] grid_features(self)
    cpdef list[str] global_features(self)

    cdef char move_object(self, unsigned int obj_id, unsigned int new_r, unsigned int new_c)
    cdef GridLocation location(self, int object_id)

    cdef GridLocation _relative_location(self, GridLocation loc, unsigned short orientation)

    cpdef void compute_observations(self,
        unsigned int[:] observer_ids,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:,:] obs)

    cpdef void step(
        self,
        unsigned int[:] actor_ids,
        unsigned int[:,:] actions,
        float[:] rewards,
        char[:] dones,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:,:] obs)

    cdef void handle_action(
        self, const Action &action,
        float *reward,
        char *done)

    cdef void _schedule_event(
        self,
        unsigned int delay,
        unsigned short event_id,
        unsigned short object_id,
        int arg)

    cdef void handle_event(self, Event event)

