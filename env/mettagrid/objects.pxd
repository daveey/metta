
from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.grid_object cimport GridObject
from puffergrid.event cimport EventHandler
from types import SimpleNamespace

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

cdef class ResetTree(EventHandler):
    pass

ctypedef GridObject[TreeProps] Tree

cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    TreeT = 2

cdef class MettaObservationEncoder(ObservationEncoder):
    pass

