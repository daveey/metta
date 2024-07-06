# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

import numpy as np

from env.puffergrid.grid_object cimport GridObject
from env.puffergrid.grid cimport PufferGrid, GridLayers

cdef enum Actions:
    ACTION_PASS = 0
    ACTION_MOVE = 1
    ACTION_ROTATE = 2
    ACTION_ATTACK = 3
    ACTION_SHIELD = 4
    ACTION_DROP = 5
    ACTION_COUNT = 6

cdef enum Objects:
    OBJECT_EMPTY = 0
    OBJECT_AGENT = 1
    OBJECT_WALL = 2
    OBJECT_TREE = 3

cdef class Tree(GridObject):
    def layer(self):
        return GridLayers.LAYER_OBJECT

    def __init__(self, id, r, c, has_food=0, cooldown=0):
        super().__init__(
            id,
            np.dtype([
                ("has_food", np.int32),
                ("cooldown", np.int32)
            ]),
            (has_food, cooldown),
            r, c
        )

cdef class Wall(GridObject):
    def layer(self):
        return GridLayers.LAYER_OBJECT

    def __init__(self, id, r, c):
        super().__init__(
            id,
            np.dtype([("hp", np.int32)]),
            (1000,),
            r, c
        )

cdef class Agent(GridObject):
    def layer(self):
        return GridLayers.LAYER_AGENT

    def __init__(self, id, r, c, agent_id, hp=100, energy=100):
        super().__init__(
            id,
            np.dtype([
                ("id", np.int32),
                ("hp", np.int32),
                ("energy", np.int32),
            ]),
            (agent_id, hp, energy),
            r, c
        )

cdef class MettaGrid(PufferGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(OBJECT_AGENT, *args, **kwargs)

    def get_object_types(self):
        return {
            "Agent": {
                "TypeId": OBJECT_AGENT,
                "Class": Agent,
            },
            "Wall": {
                "TypeId": OBJECT_WALL,
                "Class": Wall
            },
            "Tree": {
                "TypeId": OBJECT_TREE,
                "Class": Tree
            }
        }

    def print_grid(self):
        print(self.grid[0, 0, 0])
