
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdio cimport printf

from puffergrid.grid cimport Grid
from puffergrid.grid_object cimport GridObject, GridLocation
from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.grid_object cimport GridObjectBase, GridLocation, GridObjectId

cdef unsigned int GridLayer_Agent = 0
cdef unsigned int GridLayer_Object = 1

cdef struct AgentProps:
    unsigned int hp
    unsigned int energy
    unsigned int orientation

ctypedef GridObject[AgentProps] Agent

cdef struct WallProps:
    unsigned int hp

ctypedef GridObject[WallProps] Wall

cdef struct TreeProps:
    unsigned int hp
    char has_fruit

ctypedef GridObject[TreeProps] Tree

cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    TreeT = 2

cdef class MettaObservationEncoder(ObservationEncoder):
    pass

