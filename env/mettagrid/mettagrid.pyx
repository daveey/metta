
from libc.stdio cimport printf

import numpy as np
cimport numpy as cnp

from env.puffergrid.grid_env cimport GridEnv
from env.puffergrid.action cimport ActionHandler
from env.puffergrid.grid_object cimport GridObjectBase, GridLocation, GridObjectId, Orientation
from omegaconf import OmegaConf
from libcpp.vector cimport vector
from libcpp.string cimport string
from env.mettagrid.objects cimport ObjectType, Agent, Wall, Tree, GridLayer_Agent, GridLayer_Object
from env.mettagrid.objects cimport MettaObservationEncoder
from env.puffergrid.action cimport ActionHandler, ActionArg
from env.puffergrid.grid cimport Grid

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

cdef extern cppclass MoveHandler(ActionHandler):
    MoveHandler():
        pass

    char handle_action(GridObjectBase* actor, ActionArg arg, float* reward, char* done):
        # Your implementation here
        cdef unsigned short direction = arg
        if direction >= 2:
            return False
        cdef Agent* agent = <Agent*>actor
        cdef Orientation orientation = <Orientation>((agent.props.orientation + 2*(direction)) % 4)
        cdef GridLocation old_loc = agent.location
        cdef GridLocation new_loc = this._grid.relative_location(old_loc, orientation)
        if not this._grid.is_empty(new_loc.r, new_loc.c):
            return False
        cdef char s = this._grid.move_object(actor.id, new_loc)
        return s


cdef class MettaGrid(GridEnv):
    cdef:
        object _cfg
        MoveHandler _move_handler

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
