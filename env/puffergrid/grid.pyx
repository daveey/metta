# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
# distutils: language = c++

from libc.stdio cimport printf
import cython

import numpy as np
cimport numpy as cnp
from env.puffergrid.grid_object cimport GridObject, GridLocation
from env.puffergrid.grid_object cimport Orientation_Down, Orientation_Left, Orientation_Right, Orientation_Up
from libcpp.string cimport string


cdef class PufferGrid:
    def __init__(
            self,
            dict object_dtypes,
            unsigned int map_width,
            unsigned int map_height,
            unsigned short num_layers = 1,
        ):

        self._map_width = map_width
        self._map_height = map_height
        self._num_layers = num_layers

        self._current_timestep = 0

        self._np_grid = np.zeros((map_height, map_width, num_layers), dtype=np.int32)
        self._grid = self._np_grid
        self._fake_props = np.zeros(1, dtype=np.int32)
        self._fake_props_view = self._fake_props

        # self._object_types and self._objects are vectors, but their
        # 0 is not a valid index. so we add blanks to the start of the vector
        self._objects.push_back(GridObject(0, 0, GridLocation(0, 0, 0), NULL))
        self._object_data = [None]

        self._type_ids = {}
        self._object_dtypes = {}
        self._grid_features = []
        for type_name, dtype in object_dtypes.items():
            type_id = len(self._object_types)

            if type_id == 0:
                obs_offset = 0
            else:
                prev_type = self._object_types[type_id-1]
                obs_offset = prev_type.obs_offset + prev_type.num_properties

            self._type_ids[type_name] = type_id
            self._object_dtypes[type_id] = dtype
            self._object_types.push_back(TypeInfo(
                type_id = type_id,
                object_size = dtype.itemsize,
                num_properties = len(dtype.names) + 1,
                obs_offset = obs_offset
            ))
            self._grid_features.append(type_name.lower())
            self._grid_features.extend([
                type_name.lower() + ":" + p for p in dtype.names
            ])

        self._num_features = len(self._grid_features)

    cdef unsigned int _add_object(
        self,
        unsigned int object_id,
        unsigned int type_id,
        GridLocation location,
        cnp.ndarray data):

        if location.r >= 0 and location.c >= 0:
            if self._grid[location.r, location.c, location.layer] != 0:
                return 0

        object = GridObject(
            object_id,
            type_id,
            GridLocation(location.r, location.c, location.layer),
            <void *> data.data
        )
        self._objects.push_back(object)

        if location.r >= 0 and location.c >= 0:
            self._grid[location.r, location.c, location.layer] = object_id

        return object_id

    cdef GridLocation location(self, int object_id):
        return self._objects[object_id].location

    cdef char move_object(self, unsigned int obj_id, unsigned int new_r, unsigned int new_c):
        cdef GridObject *obj = &self._objects[obj_id]

        cdef unsigned int old_r = obj.location.r
        cdef unsigned int old_c = obj.location.c
        cdef unsigned int layer = obj.location.layer
        if self._grid[new_r, new_c, layer] != 0:
            return False
        obj.location.r = new_r
        obj.location.c = new_c
        self._grid[new_r, new_c, layer] = self._grid[old_r, old_c, layer]
        self._grid[old_r, old_c, layer] = 0

        return True

    cdef GridObject * _get_object(self, unsigned int obj_id):
        return &self._objects[obj_id]


    cdef GridLocation _relative_location(self, GridLocation loc, unsigned short orientation):
        if orientation == Orientation_Up:
            return GridLocation(max(0, loc.r - 1), loc.c, loc.layer)
        elif orientation == Orientation_Down:
            return GridLocation(loc.r + 1, loc.c, loc.layer)
        elif orientation == Orientation_Left:
            return GridLocation(loc.r, max(0, loc.c - 1), loc.layer)
        elif orientation == Orientation_Right:
            return GridLocation(loc.r, loc.c + 1, loc.layer)
        else:
            printf("_relative_location: Invalid orientation: %d\n", orientation)
            return GridLocation(0, 0, 0)

    cpdef void compute_observations(self,
        unsigned int[:] observer_ids,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:,:] obs):
        cdef unsigned int idx, obs_id
        cdef unsigned int num_obs = observer_ids.shape[0]
        for idx in range(num_obs):
            obs_id = observer_ids[idx]
            # printf("observing: %d\n", obs_id)
            self._compute_observation(obs_id, obs_width, obs_height, obs[idx])

    cdef void _compute_observation(
        self,
        unsigned int observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation):

        cdef unsigned int r, c, layer
        cdef int grid_r, grid_c

        cdef GridLocation observer_l = self.location(observer_id)
        cdef unsigned int observer_r = observer_l.r
        cdef unsigned int observer_c = observer_l.c
        cdef GridObject *obj
        cdef TypeInfo type_info
        cdef unsigned int obs_start, obs_end, props_end

        cdef unsigned short obs_width_r = obs_width // 2
        cdef unsigned short obs_height_r = obs_height // 2
        cdef const int[:] obj_props

        for r in range(obs_height):
            grid_r = observer_r - obs_height_r + r
            if grid_r < 0 or grid_r >= self._map_height:
                continue
            for c in range(obs_width):
                grid_c = observer_c - obs_width_r + c
                if grid_c < 0 or grid_c >= self._map_width:
                    continue

                for layer in range(self._num_layers):
                    if self._grid[grid_r, grid_c, layer] == 0:
                        continue
                    obj = self._get_object(self._grid[grid_r, grid_c, layer])
                    type_info = self._object_types[obj.type_id]

                    # location of the object presence bit
                    obs_start = type_info.obs_offset

                    # end of the observation slice
                    obs_end = obs_start + type_info.num_properties

                    props_end = type_info.num_properties - 1

                    #obj_props = <const unsigned int[:props_end]>obj.data
                    obj_props = self._fake_props_view

                    observation[obs_start, r, c] = 1
                    observation[obs_start+1:obs_end, r, c] = obj_props[0:props_end]

    cpdef void step(
        self,
        unsigned int[:] actor_ids,
        unsigned int[:,:] actions,
        float[:] rewards,
        char[:] dones,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:,:] obs):

        cdef:
            unsigned int idx
            float reward
            char done
            unsigned int num_actors = actor_ids.shape[0]
            Action action

        self._current_timestep += 1
        self._process_events()

        for idx in range(num_actors):
            reward = 0
            done = 0
            action.actor_id = actor_ids[idx]
            action.id = actions[idx][0]
            action.arg = actions[idx][1]
            action.agent_idx = idx

            self.handle_action(action, &reward, &done)

            rewards[idx] = reward
            dones[idx] = done

        # self.compute_observations(actor_ids, obs_width, obs_height, obs)

    cdef void _schedule_event(
        self,
        unsigned int delay,
        unsigned short event_id,
        unsigned short object_id,
        int arg):
        cdef Event event = Event(
            self._current_timestep + delay,
            event_id,
            object_id,
            arg
        )
        # printf("Scheduling Event: %d %d %d\n", event.timestamp, event.event_id, event.object_id)
        self._event_queue.push(event)

    cdef void _process_events(self):
        cdef Event event
        while not self._event_queue.empty():
            event = self._event_queue.top()
            if event.timestamp > self._current_timestep:
                break
            self._event_queue.pop()
            self.handle_event(event)

    ###############################
    # Subclassed Env API
    ###############################

    cdef void handle_event(self, Event event):
        printf(
            "Unhandled Event: %d %d %d\n",
            event.event_id, event.object_id, event.arg)

    cdef void handle_action(
        self,
        const Action &action,
        float *reward,
        char *done):
        printf("Unhandled Action: %d: %d(%d)\n", action.actor_id, action.id, action.arg)

    ###############################
    # Python API
    ###############################

    cpdef grid(self):
        return self._grid

    cpdef unsigned int num_features(self):
        return self._num_features

    cpdef unsigned int current_timestep(self):
        return self._current_timestep

    cpdef unsigned int map_width(self):
        return self._map_width

    cpdef unsigned int map_height(self):
        return self._map_height

    cpdef list[str] grid_features(self):
        return self._grid_features

    cpdef list[str] global_features(self):
        return []

    def type_ids(self):
        return self._type_ids

    def dtypes(self):
        return {
            name: self._object_dtypes[id]
            for name, id in self._type_ids.items()
        }

    def add_object(self, int type_id, unsigned int r=-1, unsigned int c=-1, unsigned int layer=0, **props) -> int:
        obj_data = np.zeros(1, dtype=self._object_dtypes[type_id])
        for prop_name, prop_value in props.items():
            obj_data[prop_name] = prop_value

        object_id = self._objects.size()
        self._object_data.append(obj_data)
        self._add_object(object_id, type_id, GridLocation(r, c, layer), obj_data)
        return object_id
