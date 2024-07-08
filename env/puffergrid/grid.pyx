# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

import numpy as np
cimport numpy as cnp
from env.puffergrid.grid_object cimport GridObject, GridObjectType


cdef class PufferGrid:
    def __init__(
        self,
        list[GridObjectType] object_types,
        unsigned int map_width = 32,
        unsigned int map_height = 32,
        unsigned int num_agents = 1,
        unsigned int max_timesteps = 1000,
        unsigned int obs_width = 11,
        unsigned int obs_height = 11,
        ):

        assert obs_width % 2 == 1, "obs_width must be odd"
        assert obs_height % 2 == 1, "obs_height must be odd"

        self._map_width = map_width
        self._map_height = map_height
        self._num_agents = num_agents
        self._max_timesteps = max_timesteps
        self._obs_width = obs_width
        self._obs_width_r = obs_width // 2
        self._obs_height = obs_height
        self._obs_height_r = obs_height // 2

        self._object_types = object_types
        for i, ot in enumerate(object_types):
            ot.set_id(i)

        self._layers = list(set([
            ot.grid_layer() for ot in object_types
        ]))

        self._grid_dtype = cnp.dtype([
            (l, [
                    ("object_id", np.uint32),
                    ("object_type", np.uint32)
            ]) for l in self._layers
        ] + [
                (ot.name(), [(prop, np.uint32) for prop in ot.properties()])
                for ot in object_types
        ])

        self._grid = np.zeros((map_width, map_height), dtype=self._grid_dtype)
        self._observations = np.zeros((
            num_agents,
            obs_width,
            obs_height
        ), dtype=self._grid_dtype)

        self._actions = np.zeros((num_agents, 2), dtype=np.uint8)

        self._rewards = np.zeros(num_agents, dtype=np.float32)
        self._terminals = np.zeros(num_agents, dtype=np.uint8)
        self._truncations = np.zeros(num_agents, dtype=np.uint8)
        self._masks = np.ones(num_agents, dtype=np.uint8)




        self._current_timestep = 0
        self._num_observers = 0

        self._objects = [ None ]

        self._observers = np.zeros(num_agents, dtype=np.dtype([
            ('id', np.uint32),
            ('object_id', np.uint32),
            ('r', np.uint32),
            ('c', np.uint32)
        ]))

    def observed_properties(self):
        return ["agent_id", "object_id"]

    def get_object_types(self):
        return self._object_types

    def get_grid(self):
        return self._grid

    def get_buffers(self):
        return {
            'observations': self._observations,
            'actions': self._actions,
            'rewards': self._rewards,
            'terminals': self._terminals,
            'truncations': self._truncations,
            'masks': self._masks,
        }


    def reset(self, seed=0):
        assert self._num_observers == self._num_agents, "Number of observers must match number of agents"
        self._current_timestep = 0
        self._compute_observations()

    def step(self, np_actions):
        cdef unsigned int[:] actions = np_actions
        self._compute_observations()

    def add_object(self, GridObjectType obj_type, int r=-1, int c=-1, **props):
        id = self._allocate_object_id()
        type_name = obj_type.name()

        cdef str layer = obj_type.grid_layer()
        if r >= 0 and c >= 0:
            assert self._grid[r, c][layer]["object_id"] == 0, "Cannot place object at occupied location"
            self._grid[r, c][layer]["object_type"] = obj_type.id()
            self._grid[r, c][layer]["object_id"] = id
            for prop, value in props.items():
                self._grid[r, c][type_name][prop] = value

        if obj_type.is_observer():
            assert r >= 0 and c >= 0, "Observer must be placed at a location"
            assert self._num_observers < self._num_agents, \
                f"Number of observers {self._num_observers} must match number of agents ({self._num_agents})"
            self._add_observer(id, r, c)

        return object

    def get_object(self, int object_id):
        return self._objects[object_id]

    def num_actions(self):
        return 1

    def get_layers(self):
        return self._layers

    cdef int _allocate_object_id(self):
        self._objects.append(None)
        return len(self._objects) - 1

    cdef void _add_observer(self, unsigned int id, unsigned int r, unsigned int c):
        self._observers[self._num_observers] = Observer(
            self._num_observers, id, r, c)
        self._num_observers += 1

    cdef void _compute_observations(self):
        for i in range(self._num_observers):
            observer = self._observers[i]
            self._compute_observation(observer)

    cdef void _compute_observation(self, Observer observer):
        cdef unsigned int r = observer.r
        cdef unsigned int c = observer.c
        cdef unsigned int obs_width_r = self._obs_width_r
        cdef unsigned int obs_height_r = self._obs_height_r

        #print(f"compute_observation: {observer.id} {observer.object_id} {observer.r} {observer.c}")
        cdef unsigned int r_start = max(0, r - obs_width_r)
        cdef unsigned int r_end = min(self._map_height, r + obs_height_r + 1)
        cdef unsigned int c_start = max(0, c - obs_width_r)
        cdef unsigned int c_end = min(self._map_width, c + obs_width_r + 1)
        #print("r_start", r_start, "r_end", r_end, "c_start", c_start, "c_end", c_end)

        cdef unsigned int[:, :] subarray = self._grid[r_start:r_end, c_start:c_end]

        cdef int r_offset = obs_height_r - min(obs_height_r, r)
        cdef int c_offset = obs_width_r - min(obs_width_r, c)

        self._observations[
            observer.id,
            r_offset:r_offset + subarray.shape[0],
            c_offset:c_offset + subarray.shape[1]] = subarray
