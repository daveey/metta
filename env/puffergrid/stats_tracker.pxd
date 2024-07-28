# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

cdef class StatsTracker:
    cdef:
        map[string, int] _game_stats
        vector[map[string, int]] _agent_stats

    cdef void agent_incr(
        self, unsigned int agent_idx, const char * key_str, int value)
    cdef void game_incr(self, const char * key_str, int value)

    cpdef to_pydict(self)
