from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.vector cimport vector
from env.puffergrid.action cimport ActionHandler
from env.puffergrid.event cimport EventManager
from env.puffergrid.stats_tracker cimport StatsTracker
from env.puffergrid.grid_object cimport GridObjectBase, GridObjectId, GridObject, GridLocation, Orientation, Layer
from env.puffergrid.grid cimport Grid
from env.puffergrid.event cimport EventManager
from env.puffergrid.observation_encoder cimport ObservationEncoder

from libc.stdio cimport printf

cdef class GridEnv:
    cdef:
        Grid *_grid
        EventManager *_event_manager
        vector[void*] _action_handlers
        ObservationEncoder _obs_encoder

        unsigned int _current_timestep

        list[string] _grid_features

        StatsTracker _stats
        void *_action_handler


    cdef inline void add_action_handler(self, ActionHandler *handler):
        #self._action_handlers.push_back(<void*>(handler))
        cdef void * foo = <void*>(handler)
        self._action_handler = foo
        #self._action_handler = handler
        handler.init(self._grid, self._event_manager)

    cdef inline void _compute_observation(
        self,
        GridObjectId observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation):

        cdef:
            GridLocation observer_loc = self._grid.object(observer_id).location
            GridLocation object_loc
            GridObjectBase *obj
            unsigned short obs_width_r = obs_width >> 1
            unsigned short obs_height_r = obs_height >> 1
            cdef unsigned int obs_r, obs_c

        for object_loc.r in range(observer_loc.r - obs_height_r, observer_loc.r + obs_height_r + 1):
            if object_loc.r < 0 or object_loc.r >= self._grid.height:
                continue
            for object_loc.c in range(observer_loc.c - obs_width_r, observer_loc.c + obs_width_r + 1):
                if object_loc.c < 0 or object_loc.c >= self._grid.width:
                    continue
                for object_loc.layer in range(self._grid.num_layers):
                    obj = self._grid.object_at(object_loc)
                    if obj == NULL:
                        continue

                    obs_r = object_loc.r - (observer_loc.r - obs_height_r)
                    obs_c = object_loc.c - (observer_loc.c - obs_width_r)
                    printf("encoding object at (%d, %d, %d) to (%d, %d, %d)\n", object_loc.r, object_loc.c, object_loc.layer, obs_r, obs_c, object_loc.layer)
                    self._obs_encoder.encode(obj, observation[obs_r, obs_c])


    cpdef grid(self)
    cpdef unsigned int current_timestep(self)
    cpdef unsigned int num_features(self)
    cpdef unsigned int map_width(self)
    cpdef unsigned int map_height(self)
    cpdef list[str] grid_features(self)

    cpdef void reset(
        self,
        GridObjectId[:] actor_ids,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:,:] obs)

    cpdef void step(
        self,
        GridObjectId[:] actor_ids,
        unsigned int[:,:] actions,
        float[:] rewards,
        char[:] dones,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:,:] obs)


