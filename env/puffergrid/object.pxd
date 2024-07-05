# object.pxd
cdef class GridObject:
    cdef:
        int r
        int c
        int layer
        int id
