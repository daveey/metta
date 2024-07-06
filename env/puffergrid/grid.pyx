# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

import numpy as np
cimport numpy as cnp

cdef enum Objects:
    OBJECT_AGENT = 1

cdef class PufferGrid:
    def __init__(
        self,
        observer_type_id,
        map_width = 32,
        map_height = 32,
        num_agents = 1,
        int max_timesteps = 1000,
        obs_width = 11,
        obs_height = 11,
        ):

        assert obs_width % 2 == 1, "obs_width must be odd"
        assert obs_height % 2 == 1, "obs_height must be odd"

        self._grid = np.zeros(
            (GridLayers.LAYER_COUNT, map_width, map_height),
            dtype=np.uint32
        )
        self._observations = np.zeros((
            num_agents,
            len(self.observed_properties()),
            obs_width,
            obs_height
        ), dtype=np.uint32)
        self._actions = np.zeros((num_agents, 2), dtype=np.uint8)

        self._rewards = np.zeros(num_agents, dtype=np.float32)
        self._terminals = np.zeros(num_agents, dtype=np.uint8)
        self._truncations = np.zeros(num_agents, dtype=np.uint8)
        self._masks = np.ones(num_agents, dtype=np.uint8)

        self._map_width = map_width
        self._map_height = map_height
        self._num_agents = num_agents
        self._max_timesteps = max_timesteps
        self._obs_width = obs_width
        self._obs_height = obs_height

        self._current_timestep = 0
        self._num_observers = 0

        self._objects = [ None ]

        self._observer_type_id  = observer_type_id
        self._observers = np.zeros(num_agents, dtype=np.dtype([
            ('id', np.uint32),
            ('object_id', np.uint32),
            ('r', np.uint32),
            ('c', np.uint32)
        ]))

        object_types = self.get_object_types()

        self._object_classes = {
            obj_type["TypeId"]: obj_type["Class"] for obj_type in object_types.values()
        }

    def observed_properties(self):
        return ["agent_id", "object_id"]

    def get_object_types(self):
        return {}

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

    def add_object(self, int obj_type, int r=-1, int c=-1, **props):
        id = self._allocate_object_id()

        object = self._object_classes[obj_type](id, r, c, **props)
        cdef int layer = object.layer()
        self._objects[id] = object
        # print(f"make_object: {id} ({layer}, {r}, {c}) {obj_type} {props}")
        if r >= 0 and c >= 0:
            assert self._grid[layer, r, c] == 0, "Cannot place object at occupied location"
            self._grid[layer, r, c] = id

        if obj_type == self._observer_type_id:
            self._add_observer(id, r, c)

        return object

    def get_object(self, int object_id):
        return self._objects[object_id]

    def num_actions(self):
        return 1

    cdef int _allocate_object_id(self):
        self._objects.append(None)
        return len(self._objects) - 1

    cdef void _add_observer(self, unsigned int id, unsigned int r, unsigned int c):
        assert r >= 0 and c >= 0, "Observer must be placed at a location"
        self._observers[self._num_observers] = Observer(
            self._num_observers, id, r, c)
        self._num_observers += 1

    cdef void _compute_observations(self):
        for i in range(self._num_observers):
            observer = self._observers[i]
            self._compute_observation(observer)

    cdef void _compute_observation(self, Observer observer):
        print(f"compute_observation: {observer.id} {observer.object_id} {observer.r} {observer.c}")

