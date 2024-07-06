# object.pxd
cimport numpy as cnp

cdef struct GridLocation:
    unsigned int r
    unsigned int c

cdef class GridObject:
    cdef:
        unsigned int _id
        GridLocation _location
        cnp.dtype _props_dtype
        cnp.ndarray _props

