# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
# distutils: language = c++

# Include grid.h for Orientation definitions
cdef extern from "grid.h":
    cdef unsigned short Orientation_Up
    cdef unsigned short Orientation_Down
    cdef unsigned short Orientation_Left
    cdef unsigned short Orientation_Right

from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as cnp

from env.puffergrid.grid_object cimport GridLocation


from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef struct GridLocation:
    unsigned int r
    unsigned int c
    unsigned short layer

cdef struct GridObject:
    unsigned int id
    unsigned short type_id
    GridLocation location
    void * data

