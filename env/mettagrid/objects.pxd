# distutils: language=c++
# cython: warn.undeclared=False

cimport cython

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from puffergrid.grid_env import StatsTracker
from libc.stdio cimport printf
from puffergrid.observation_encoder cimport ObservationEncoder, ObsType
from puffergrid.grid_object cimport GridObject, TypeId, GridCoord, GridLocation, GridObjectId
from puffergrid.event cimport EventHandler, EventArg

cdef enum GridLayer:
    Agent_Layer = 0
    Object_Layer = 1

ctypedef map[string, int] ObjectConfig

cdef cppclass MettaObject(GridObject):
    unsigned int hp

    inline void init_mo(ObjectConfig cfg):
        this.hp = cfg[b"hp"]

    inline char usable(const Agent *actor):
        return False

    inline char attackable():
        return False

cdef cppclass Usable(MettaObject):
    unsigned int use_cost
    unsigned int cooldown
    unsigned char ready

    inline void init_usable(ObjectConfig cfg):
        this.use_cost = cfg[b"use_cost"]
        this.cooldown = cfg[b"cooldown"]
        this.ready = 1

    inline char usable(const Agent *actor):
        return this.ready and this.use_cost <= actor.energy

cdef enum ObjectType:
    AgentT = 0
    WallT = 1
    GeneratorT = 2
    ConverterT = 3
    AltarT = 4
    Count = 5

cdef vector[string] ObjectTypeNames # defined in objects.pyx

cdef enum InventoryItem:
    r1 = 0,
    r2 = 1,
    r3 = 2,
    InventoryCount = 3

cdef vector[string] InventoryItemNames # defined in objects.pyx


cdef cppclass Agent(MettaObject):
    char frozen
    unsigned int energy
    unsigned int orientation
    char shield
    vector[unsigned short] inventory

    inline Agent(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.AgentT, GridLocation(r, c, GridLayer.Agent_Layer))
        MettaObject.init_mo(cfg)
        this.frozen = False
        this.energy = cfg[b"initial_energy"]
        this.orientation = 0
        this.inventory.resize(InventoryItem.InventoryCount)

    inline void update_inventory(InventoryItem item, short amount):
        this.inventory[<InventoryItem>item] += amount

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.frozen
        obs[3] = this.energy
        obs[4] = this.orientation
        obs[5] = this.shield

        cdef unsigned short idx = 6
        cdef unsigned short i
        for i in range(InventoryItem.InventoryCount):
            obs[idx + i] = this.inventory[i]

    @staticmethod
    inline vector[string] feature_names():
        return [
            "agent", "agent:hp", "agent:frozen", "agent:energy", "agent:orientation",
            "agent:shield"
        ] + [
            "agent:inv:" + n for n in InventoryItemNames]

cdef cppclass Wall(MettaObject):
    inline Wall(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.WallT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]

cdef cppclass Generator(Usable):
    unsigned int r1

    inline Generator(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.GeneratorT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)
        this.r1 = cfg[b"initial_resources"]

    inline char usable(const Agent *actor):
        return Usable.usable(actor) and this.r1 > 0

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.r1
        obs[3] = this.ready


    @staticmethod
    inline vector[string] feature_names():
        return ["generator", "generator:hp", "generator:r1", "generator:ready"]

cdef cppclass Converter(Usable):
    InventoryItem input_resource
    InventoryItem output_resource
    short output_energy

    inline Converter(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.ConverterT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)
        this.input_resource = InventoryItem.r1
        this.output_resource = InventoryItem.r2
        this.output_energy = cfg[b"energy_output.r1"]

    inline char usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[this.input_resource] > 0

    inline obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = input_resource
        obs[3] = output_resource
        obs[4] = output_energy
        obs[5] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["converter", "converter:hp", "converter:input_resource", "converter:output_resource", "converter:output_energy", "converter:ready"]

cdef cppclass Altar(Usable):
    inline Altar(GridCoord r, GridCoord c, ObjectConfig cfg):
        GridObject.init(ObjectType.AltarT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(cfg)
        Usable.init_usable(cfg)

    inline void obs(ObsType[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["altar", "altar:hp", "altar:ready"]

cdef map[TypeId, GridLayer] ObjectLayers

cdef class ResetHandler(EventHandler):
    cdef inline void handle_event(self, GridObjectId obj_id, EventArg arg):
        cdef Usable *usable = <Usable*>self.env._grid.object(obj_id)
        usable.ready = True
        self.env._stats.game_incr("resets." + ObjectTypeNames[usable._type_id])

cdef enum Events:
    Reset = 0

cdef class MettaObservationEncoder(ObservationEncoder):
    cdef vector[short] _offsets
    cdef vector[string] _feature_names

