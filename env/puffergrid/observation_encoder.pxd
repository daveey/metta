from libcpp.vector cimport vector
from libcpp.string cimport string

from env.puffergrid.grid_object cimport GridObjectBase

cdef class ObservationEncoder:

    cdef encode(self, const GridObjectBase *obj, int[:] obs)

    @staticmethod
    cdef vector[string] feature_names()
