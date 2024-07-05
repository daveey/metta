# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

cdef class GridObject:
    def __init__(self, int id, int r, int c, int layer):
        self.id = id
        self.r = r
        self.c = c
        self.layer = layer
