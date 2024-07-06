# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
import numpy as np

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
