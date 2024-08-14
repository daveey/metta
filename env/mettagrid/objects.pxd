# distutils: language=c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from puffergrid.grid_env import StatsTracker
from libc.stdio cimport printf

from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.grid_object cimport GridObject, GridCoord, GridLocation
from puffergrid.event cimport EventHandler

cdef enum GridLayer:
    Agent_Layer = 0
    Object_Layer = 1

cdef cppclass MettaObject(GridObject):

    inline char usable():
        return False
    inline char attackable():
        return False

cdef cppclass Attackable:
    unsigned int hp
    inline char attackable():
        return True

cdef cppclass Usable:
    unsigned int energy_cost
    unsigned int cooldown
    unsigned char ready

    inline char usable():
        return True
    inline char on_use():
        printf("object used\n")

cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    GeneratorT = 2
    ConverterT = 3
    AltarT = 4
    Count = 5
cdef cppclass Agent(MettaObject, Attackable):
    unsigned int hp
    unsigned int energy
    unsigned int orientation

    inline Agent(GridCoord r, GridCoord c):
        init(ObjectType.AgentT, GridLocation(r, c, GridLayer.Agent_Layer))
        hp = 1
        energy = 100
        orientation = 0

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = energy
        obs[3] = orientation

    @staticmethod
    inline vector[string] feature_names():
        return ["agent", "agent:hp", "agent:energy", "agent:orientation"]

cdef cppclass Wall(MettaObject, Attackable):
    inline Wall(GridCoord r, GridCoord c):
        init(ObjectType.WallT, GridLocation(r, c, GridLayer.Object_Layer))
        hp = 111

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]

cdef cppclass Generator(MettaObject, Attackable, Usable):
    unsigned int r1

    inline Generator(GridCoord r, GridCoord c):
        init(ObjectType.GeneratorT, GridLocation(r, c, GridLayer.Object_Layer))
        r1 = 10
        ready = 1

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = r1
        obs[3] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["generator", "generator:hp", "generator:r1", "generator:ready"]

cdef cppclass Converter(MettaObject, Attackable, Usable):
    char input_resource
    char output_resource
    char output_energy

    inline Converter(GridCoord r, GridCoord c):
        init(ObjectType.ConverterT, GridLocation(r, c, GridLayer.Object_Layer))

    inline obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = input_resource
        obs[3] = output_resource
        obs[4] = output_energy
        obs[5] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["converter", "converter:hp", "converter:input_resource", "converter:output_resource", "converter:output_energy", "converter:ready"]

cdef cppclass Altar(MettaObject, Attackable, Usable):
    inline Altar(GridCoord r, GridCoord c):
        init(ObjectType.AltarT, GridLocation(r, c, GridLayer.Object_Layer))
        hp = 10
        ready = 1

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["altar", "altar:hp", "altar:ready"]


cdef class ResetTreeHandler(EventHandler):
    pass

cdef enum Events:
    ResetTree = 0

cdef class MettaObservationEncoder(ObservationEncoder):
    cdef vector[short] _offsets
    cdef vector[string] _feature_names

