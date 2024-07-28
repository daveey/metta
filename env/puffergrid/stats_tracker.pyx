# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
# distutils: language = c++

from libc.stdio cimport printf

from libcpp.string cimport string

cdef class StatsTracker:
    def __init__(self, unsigned int num_agents) -> None:
        self._agent_stats.resize(num_agents)

    cdef void game_incr(self, const char * key_str, int value):
        cdef string key = string(key_str)
        self._game_stats[key] += value

    cdef void agent_incr(
        self, unsigned int agent_idx, const char * key_str, int value):
        cdef string key = string(key_str)
        self._agent_stats[agent_idx][key] += value

    cpdef to_pydict(self):
        agent_stat_names = set()
        new_agent_stats = []
        for agent_stats in self._agent_stats:
            new_stats = {}
            for k, v in agent_stats:
                agent_stat_names.add(k)
                new_stats[k] = v
            new_agent_stats.append(new_stats)

        # We have to convert stat names to unicode strings
        # for better python interface. We also want to make
        # sure to return 0s for any missing stats otherwise
        # pufferlib won't average correctly.
        return {
            "game_stats": {
                k.decode(): v for k, v in self._game_stats
            },
            "agent_stats": [{
                    k.decode(): a.get(k, 0) for k in agent_stat_names
                } for a in new_agent_stats
            ]
        }
