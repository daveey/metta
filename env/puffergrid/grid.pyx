# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
# distutils: language = c++

from libc.stdio cimport printf

# Include grid.h for Orientation definitions
cdef extern from "grid.h":
    pass

import numpy as np
cimport numpy as cnp
from env.puffergrid.grid_object cimport GridObject, GridLocation
from env.puffergrid.grid_object cimport Orientation_Down, Orientation_Left, Orientation_Right, Orientation_Up

cdef class PufferGrid:
    def __init__(
        self,
        dict object_types,
        unsigned int map_width = 32,
        unsigned int map_height = 32,
        unsigned int max_timesteps = 1000,
        unsigned short num_layers = 1
        ):

        self._map_width = map_width
        self._map_height = map_height
        self._max_timesteps = max_timesteps
        self._num_layers = num_layers

        self._current_timestep = 0

        self._np_grid = np.zeros((map_width, map_height, num_layers), dtype=np.uint32)
        self._grid = self._np_grid

        # self._object_types and self._objects are vectors, but their
        # 0 is not a valid index. so we add blanks to the start of the vector
        self._object_types.push_back(TypeInfo(0, 0, 0, 0, 0))
        self._objects.push_back(GridObject(0, 0, GridLocation(0, 0), NULL))
        self._object_data = [None]

        self._type_ids = {}
        self._object_dtypes = {}
        for type_name, type_info in object_types.items():
            dtype = type_info[1]
            layer = type_info[2]
            type_id = len(self._object_types)
            prev_type = self._object_types[type_id-1]
            obs_offset = prev_type.obs_offset + prev_type.num_properties + 1
            self._type_ids[type_name] = type_id
            self._object_dtypes[type_id] = dtype
            self._object_types.push_back(TypeInfo(
                type_id = type_id,
                object_size = dtype.itemsize,
                grid_layer = layer,
                num_properties = len(dtype.fields),
                obs_offset = obs_offset
            ))
        self._num_features = 100


    def get_types(self):
        return self._type_ids

    def get_dtypes(self):
        return {
            name: self._object_dtypes[id]
            for name, id in self._type_ids.items()
        }

    cpdef get_grid(self):
        return self._grid

    cpdef unsigned int get_num_features(self):
        return self._num_features

    cpdef unsigned int get_current_timestep(self):
        return self._current_timestep

    def num_actions(self):
        return 1

    def add_object(self, int type_id, unsigned int r=-1, unsigned int c=-1, **props) -> int:
        obj_data = np.zeros(1, dtype=self._object_dtypes[type_id])
        for prop_name, prop_value in props.items():
            obj_data[prop_name] = prop_value

        object_id = self._objects.size()
        type_info = self._object_types[type_id]
        self._object_data.append(obj_data)
        self._add_object(object_id, type_info, r, c, obj_data)
        return object_id

    cdef unsigned int _add_object(
        self,
        unsigned int object_id,
        TypeInfo type_info,
        unsigned int r, unsigned int c,
        cnp.ndarray data):

        object = GridObject(
            object_id,
            type_info.type_id,
            GridLocation(r, c),
            <void *> data.data
        )
        self._objects.push_back(object)
        cdef unsigned short grid_layer = type_info.grid_layer

        if r >= 0 and c >= 0:
            assert self._grid[r, c, grid_layer] == 0, "Cannot place object at occupied location"
            self._grid[r, c, grid_layer] = object_id

        return object_id

    def get_object(self, object_id):
        obj = self._objects[object_id]
        obj_data = self._object_data[object_id]
        return {
            "object_id": object_id,
            "type_id": obj.type_id,
            "r": obj.location.r,
            "c": obj.location.c,
            "data": obj_data
        }

    cpdef GridLocation location(self, int object_id):
        return self._objects[object_id].location

    cpdef void move_object(self, unsigned int obj_id, unsigned int new_r, unsigned int new_c):
        cdef GridObject obj = self._objects[obj_id]
        cdef unsigned int old_r = obj.location.r
        cdef unsigned int old_c = obj.location.c
        cdef unsigned short grid_layer = self._object_types[obj.type_id].grid_layer
        obj.location.r = new_r
        obj.location.c = new_c
        self._grid[new_r, new_c, grid_layer] = self._grid[old_r, old_c, grid_layer]
        self._grid[old_r, old_c, grid_layer] = 0

    cdef GridObject _get_object(self, unsigned int obj_id):
        return self._objects[obj_id]


    cdef GridLocation _relative_location(self, GridLocation loc, unsigned short orientation):
        if orientation == Orientation_Up:
            return GridLocation(max(0, loc.r - 1), loc.c)
        elif orientation == Orientation_Down:
            return GridLocation(loc.r + 1, loc.c)
        elif orientation == Orientation_Left:
            return GridLocation(loc.r, max(0, loc.c - 1))
        elif orientation == Orientation_Right:
            return GridLocation(loc.r, loc.c + 1)
        else:
            raise ValueError(f"Invalid orientation: {orientation}")

    cpdef void compute_observations(self,
        unsigned int[:] observer_ids,
        unsigned short obs_width,
        unsigned short obs_height,
        unsigned int[:,:,:,:] obs):
        cdef unsigned int idx, obs_id
        cdef unsigned int num_obs = len(observer_ids)
        for idx in range(num_obs):
            obs_id = observer_ids[idx]
            # printf("observing: %d\n", obs_id)
            self._compute_observation(obs_id, obs_width, obs_height, obs[idx])

    cdef void _compute_observation(
        self,
        unsigned int observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        unsigned int[:,:,:] observation):

        cdef unsigned int r, c, layer
        cdef int grid_r, grid_c

        cdef GridObject observer = self._objects[observer_id]
        cdef unsigned int observer_r = observer.location.r
        cdef unsigned int observer_c = observer.location.c
        cdef GridObject obj
        cdef TypeInfo type_info
        cdef unsigned int obs_start, obs_end, props_end

        cdef unsigned short obs_width_r = obs_width // 2
        cdef unsigned short obs_height_r = obs_height // 2
        cdef const unsigned int[::1] obj_props

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
                    obj = self._objects[self._grid[grid_r, grid_c, layer]]
                    type_info = self._object_types[obj.type_id]
                    props_end = type_info.num_properties + 1
                    obs_start = type_info.obs_offset
                    obs_end = obs_start + props_end + 1
                    # printf("observing: %d %d %d %d\n", r, c, obs_start, obs_end)

                    obj_props = <const unsigned int[:props_end]>obj.data

                    observation[r, c, obs_start] = 1
                    observation[r, c, obs_start+1:obs_end] = obj_props[0:props_end]

    cpdef void step(
        self,
        unsigned int[:] actor_ids,
        unsigned int[:,:] actions,
        float[:] rewards,
        char[:] dones):
        cdef:
            unsigned int idx
            float reward = 0
            char done = 0
            unsigned int num_actors = actor_ids.shape[0]

        self._current_timestep += 1
        self._process_events()

        for idx in range(num_actors):
            self.handle_action(
                actor_ids[idx],
                actions[idx][0], actions[idx][1],
                &reward, &done)

            rewards[idx] = reward
            dones[idx] = done

    cdef void handle_action(
        self,
        unsigned int actor_id,
        unsigned short action_id,
        unsigned short action_arg,
        float *reward,
        char *done):
        printf("Unhandled Action: %d %d %d\n", actor_id, action_id, action_arg)

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
        self._event_queue.push(event)

    cdef void _process_events(self):
        while not self._event_queue.empty() and self._event_queue.top().timestamp <= self._current_timestep:
            event = self._event_queue.top()
            self._event_queue.pop()
            self.handle_event(event)

    cdef void handle_event(self, Event event):
        printf(
            "Unhandled Event: %d %d %d\n",
            event.event_id, event.object_id, event.arg)
