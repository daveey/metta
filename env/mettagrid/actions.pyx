
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


cdef cppclass MoveHandler(ActionHandler):

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

