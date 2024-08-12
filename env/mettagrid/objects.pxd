# distutils: language=c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from puffergrid.grid_env import StatsTracker
from libc.stdio cimport printf

from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.grid_object cimport GridObject
from puffergrid.event cimport EventHandler

cdef enum GridLayer:
    Agent_Layer = 0
    Object_Layer = 1

cdef cppclass MettaObjectProps:
    inline char usable():
        return False
    inline char attackable():
        return False
ctypedef GridObject[MettaObjectProps] MettaObject

cdef cppclass Attackable(MettaObjectProps):
    unsigned int hp
    inline char attackable():
        return True

cdef cppclass UsableProps(MettaObjectProps):
    unsigned int energy_cost
    unsigned int cooldown
    unsigned char ready

    inline char usable():
        return True
    inline char on_use():
        printf("object used\n")

ctypedef GridObject[UsableProps] UsableObject

cdef cppclass AgentProps(Attackable):
    unsigned int hp
    unsigned int energy
    unsigned int orientation

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = energy
        obs[3] = orientation

    @staticmethod
    inline vector[string] feature_names():
        return ["agent", "agent:hp", "agent:energy", "agent:orientation"]

ctypedef GridObject[AgentProps] Agent

cdef cppclass WallProps(Attackable):
    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]

ctypedef GridObject[WallProps] Wall

cdef cppclass GeneratorProps(Attackable, UsableProps):
    unsigned int r1

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = r1
        obs[3] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["generator", "generator:hp", "generator:r1", "generator:ready"]

ctypedef GridObject[GeneratorProps] Generator

cdef cppclass ConverterProps(Attackable, UsableProps):
    char input_resource
    char output_resource
    char output_energy

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

ctypedef GridObject[ConverterProps] Converter

cdef cppclass AltarProps(Attackable, UsableProps):
    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["altar", "altar:hp", "altar:ready"]

ctypedef GridObject[AltarProps] Altar


cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    GeneratorT = 2
    ConverterT = 3
    AltarT = 4
    Count = 5

cdef class ResetTreeHandler(EventHandler):
    pass

cdef enum Events:
    ResetTree = 0

cdef class MettaObservationEncoder(ObservationEncoder):
    cdef vector[short] _offsets
    cdef vector[string] _feature_names

