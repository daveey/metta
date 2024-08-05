
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
from puffergrid.action cimport ActionHandler, ActionArg
from puffergrid.grid cimport Grid


cdef cppclass MoveHandler(ActionHandler):
    char handle_action(GridObjectBase* actor, ActionArg arg, float* reward, char* done)
