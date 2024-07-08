# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
import numpy as np


cdef class GridObjectType:
    def __init__(self, name, grid_layer, properties, is_observer=False):
        self._name = name
        self._id = -1
        self._grid_layer = grid_layer
        self._properties = properties
        self._is_observer = is_observer

    def id(self):
        return self._id

    def set_id(self, int _id):
        self._id = _id

    def name(self):
        return self._name

    def grid_layer(self):
        return self._grid_layer

    def properties(self):
        return self._properties

    def is_observer(self):
        return self._is_observer

cdef class GridObject():

    def __init__(
        self, int id, props_dtype, props,
        int r = -1, int c = -1):
        self._id = id
        self._props_dtype = props_dtype
        self._props = np.array(props, dtype=self._props_dtype)
        self._location = GridLocation(r, c)

    def id(self):
        return self._id

    def props(self):
        return self._props

    def location(self):
        return self._location

    def layer(self):
        return -1
