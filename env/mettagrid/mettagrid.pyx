# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

import numpy as np

from env.puffergrid.grid_object cimport GridObject
from env.puffergrid.grid cimport PufferGrid, GridObjectType

cdef class MettaGrid(PufferGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(
            object_types=[
                GridObjectType(
                    name="Agent",
                    grid_layer="agents",
                    properties=[
                        "id",
                        "hp",
                        "energy"
                    ],
                    is_observer=True
                ),
                GridObjectType(
                    name="Wall",
                    grid_layer="objects",
                    properties=[ "hp" ]
                ),
                GridObjectType(
                    name="Tree",
                    grid_layer="objects",
                    properties=[
                        "has_food",
                        "cooldown"
                    ]
                )
            ],
            *args, **kwargs
        )




cdef enum Actions:
    ACTION_PASS = 0
    ACTION_MOVE = 1
    ACTION_ROTATE = 2
    ACTION_ATTACK = 3
    ACTION_SHIELD = 4
    ACTION_DROP = 5
    ACTION_COUNT = 6
