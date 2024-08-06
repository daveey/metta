from libcpp.vector cimport vector
from libcpp.string cimport string

from puffergrid.grid_object cimport GridObjectBase

cdef class ObservationEncoder:

    cdef encode(self, const GridObjectBase *obj, int[:] obs)

    cdef vector[string] feature_names(self)