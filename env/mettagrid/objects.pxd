# distutils: language=c++
from libcpp.vector cimport vector
from libcpp.string cimport string

from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.grid_object cimport GridObject
from puffergrid.event cimport EventHandler

cdef unsigned int GridLayer_Agent = 0
cdef unsigned int GridLayer_Object = 1

cdef cppclass AgentProps:
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

cdef cppclass WallProps:
    unsigned int hp

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]

ctypedef GridObject[WallProps] Wall

cdef cppclass GeneratorProps:
    unsigned int hp
    unsigned int r1
    char ready

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = r1
        obs[3] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["generator", "generator:hp", "generator:r1", "generator:ready"]

ctypedef GridObject[GeneratorProps] Generator

cdef cppclass ConverterProps:
    unsigned int hp
    char input_resource
    char output_resource
    char output_energy
    char ready

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

cdef cppclass AltarProps:
    unsigned int hp
    char ready

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

