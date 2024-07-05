import numpy as np
cimport numpy as cnp

cdef class PufferGrid:
    def __init__(
        self,
        map_width = 32,
        map_height = 32,
        num_agents = 1,
        int max_timesteps = 1000,
        obs_width = 11,
        obs_height = 11,
        ):

        assert obs_width % 2 == 1, "obs_width must be odd"
        assert obs_height % 2 == 1, "obs_height must be odd"

        self.grid = np.zeros(
            (GridLayers.LAYER_COUNT, map_width, map_height),
            dtype=np.uint32
        )
        self.observations = np.zeros((
            num_agents,
            len(self.observed_properties()),
            obs_width,
            obs_height
        ), dtype=np.uint32)

        self.rewards = np.zeros(num_agents, dtype=np.float32)
        self.terminals = np.zeros(num_agents, dtype=np.uint8)
        self.truncations = np.zeros(num_agents, dtype=np.uint8)
        self.masks = np.ones(num_agents, dtype=np.uint8)

        self.map_width = map_width
        self.map_height = map_height
        self.num_agents = num_agents
        self.max_timesteps = max_timesteps
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.current_timestep = 0

        self.agents = []
        self.objects = [ None ]

    def observed_properties(self):
        return ["agent_id", "object_id"]

    def get_grid(self):
        return self.grid

    def get_buffers(self):
        return {
            'observations': self.observations,
            'rewards': self.rewards,
            'terminals': self.terminals,
            'truncations': self.truncations,
            'masks': self.masks,
        }

    cdef void _compute_observations(self):
        pass

    def reset(self, observations, seed=0):
        self.current_timestep = 0
        self.observations = observations # why?
        self._compute_observations()

    def step(self, np_actions):
        cdef unsigned int[:] actions = np_actions
        self._compute_observations()

    cdef int _allocate_object_id(self):
        self.objects.append(None)
        return len(self.objects) - 1

    def num_actions(self):
        return 1
