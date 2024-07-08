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

cdef class GridObjectType:
    cdef:
        unsigned int _id
        str _name
        str _grid_layer
        list _properties
        char _is_observer
