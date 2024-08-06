
from libc.stdio cimport printf

import numpy as np
cimport numpy as cnp

from puffergrid.grid_env cimport GridEnv
from puffergrid.action cimport ActionHandler
from puffergrid.grid_object cimport GridObjectBase, GridLocation, GridObjectId, Orientation
from omegaconf import OmegaConf
from libcpp.vector cimport vector
from libcpp.string cimport string
from env.mettagrid.objects cimport ObjectType, Agent, Wall, Tree, GridLayer_Agent, GridLayer_Object
from env.mettagrid.objects cimport MettaObservationEncoder
from puffergrid.grid cimport Grid
from env.mettagrid.actions cimport MoveHandler
from puffergrid.action cimport ActionHandler, ActionArg
ObjectLayers = {
    ObjectType.AgentT: GridLayer_Agent,
    ObjectType.WallT: GridLayer_Object,
    ObjectType.TreeT: GridLayer_Object
}

ObjectBlocks = {
    ObjectType.AgentT: GridLayer_Agent,
    ObjectType.WallT: [GridLayer_Object, GridLayer_Agent],
    ObjectType.TreeT: [GridLayer_Object, GridLayer_Agent]
}

cdef class MettaGrid(GridEnv):
    cdef:
        object _cfg

    def __init__(self, cfg: OmegaConf, map: np.ndarray):
        self._cfg = cfg

        GridEnv.__init__(
            self,
            map.shape[0],
            map.shape[1],
            ObjectLayers.values(),
            11,11,
            MettaObservationEncoder(),
            [
                MoveHandler(),
            ]
        )

    cpdef make_agent(self, row: int, col: int):
        if not self._grid.is_empty(row, col):
            return -1

        cdef Agent *agent = self._grid.create_object[Agent](ObjectType.AgentT, row, col)
        if agent == NULL:
            return -1

        agent.props.hp = 100
        agent.props.energy = 100
        agent.props.orientation = 0
        print("Agent: id: ", agent.id, " hp: ", agent.props.hp, " energy: ", agent.props.energy, " orientation: ", agent.props.orientation)

        self.add_agent(agent)

        return agent.id

    cpdef make_wall(self, row: int, col: int):
        if not self._grid.is_empty(row, col):
            return -1

        cdef Wall *wall = self._grid.create_object[Wall](ObjectType.WallT, row, col)
        if wall == NULL:
            print("Failed to create wall")
            return -1

        wall.props.hp = 100
        return wall.id

    cpdef make_tree(self, row: int, col: int):
        if not self._grid.is_empty(row, col):
            return -1

        cdef Tree *tree = self._grid.create_object[Tree](ObjectType.TreeT, row, col)
        if tree == NULL:
            return -1

        tree.props.hp = 100
        tree.props.has_fruit = 1
        return tree.id

