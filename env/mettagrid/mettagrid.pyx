
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
from env.mettagrid.actions cimport MoveHandler
from puffergrid.grid cimport Grid
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
            MettaObservationEncoder()
        )
        cdef ActionHandler *move_handler = new MoveHandler()
        move_handler.init(self._grid, self._event_manager)
        cdef void *handler = <void*>move_handler
        self.add_action_handler(move_handler)

    cpdef make_agent(self, row: int, col: int):
        if not self._grid.is_empty(row, col):
            print("cell is not empty")
            return -1

        cdef Agent *agent = self._grid.create_object[Agent](ObjectType.AgentT, row, col)
        if agent == NULL:
            print("Failed to create agent")
            return -1

        agent.props.hp = 100
        agent.props.energy = 100
        agent.props.orientation = 0
        print("Agent: id: ", agent.id, " hp: ", agent.props.hp, " energy: ", agent.props.energy, " orientation: ", agent.props.orientation)
        return agent.id

    cpdef make_wall(self, row: int, col: int):
        if not self._grid.is_empty(row, col):
            print("cell is not empty")
            return -1

        cdef Wall *wall = self._grid.create_object[Wall](ObjectType.WallT, row, col)
        if wall == NULL:
            print("Failed to create wall")
            return -1

        wall.props.hp = 100
        return wall.id

    cpdef make_tree(self, row: int, col: int):
        if not self._grid.is_empty(row, col):
            print("cell is not empty")
            return -1

        cdef Tree *tree = self._grid.create_object[Tree](ObjectType.TreeT, row, col)
        if tree == NULL:
            print("Failed to create tree")
            return -1

        tree.props.hp = 100
        tree.props.has_fruit = 1
        return tree.id

    cpdef compute_observation(
        self,
        GridObjectId observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation):

        self._compute_observation(observer_id, obs_width, obs_height, observation)
