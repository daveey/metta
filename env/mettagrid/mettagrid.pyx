# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

from env.puffergrid.object cimport GridObject
from env.puffergrid.grid cimport PufferGrid, GridLayers

cdef enum Actions:
    ACTION_PASS = 0
    ACTION_MOVE = 1
    ACTION_ROTATE = 2
    ACTION_ATTACK = 3
    ACTION_SHIELD = 4
    ACTION_DROP = 5
    ACTION_COUNT = 6


cdef class Wall(GridObject):
    def __init__(self, int id, int r, int c):
        super().__init__(id, r, c, GridLayers.LAYER_OBJECT)

cdef class Agent(GridObject):
    def __init__(self, int id,  int r, int c):
        super().__init__(id, r, c, GridLayers.LAYER_AGENT)

cdef class MettaGrid(PufferGrid):

    def place_agent(self, r, c):
        agent_id = self._allocate_object_id()
        agent = Agent(agent_id, r, c)
        self.objects[agent_id] = agent
        self.grid[agent.layer, agent.r, agent.c] = agent.id

    def place_wall(self, r, c):
        wall_id = self._allocate_object_id()
        wall = Wall(wall_id, r, c)
        self.objects[wall_id] = wall
        self.grid[wall.layer, wall.r, wall.c] = wall.id

    def print_grid(self):
        print(self.grid[0, 0, 0])
