# distutils: language=c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from puffergrid.grid_env import StatsTracker
from libc.stdio cimport printf

from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.grid_object cimport GridObject, TypeId, GridCoord, GridLocation, GridObjectId
from puffergrid.event cimport EventHandler, EventArg
cdef enum GridLayer:
    Agent_Layer = 0
    Object_Layer = 1

cdef cppclass MettaObject(GridObject):
    unsigned int hp

    inline void init_mo(unsigned int hp):
        this.hp = hp

    inline char usable(const Agent *actor):
        return False

    inline char attackable():
        return False

cdef cppclass Usable(MettaObject):
    unsigned int energy_cost
    unsigned int cooldown
    unsigned char ready

    inline void init_usable(unsigned int energy_cost, unsigned int cooldown):
        this.energy_cost = energy_cost
        this.cooldown = cooldown
        this.ready = 1

    inline char usable(const Agent *actor):
        return this.ready and this.energy_cost <= actor.energy

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
    unsigned int hp
    unsigned int energy
    unsigned int orientation
    char shield
    char energy_upkeep
    vector[unsigned short] inventory

    inline Agent(GridCoord r, GridCoord c):
        GridObject.init(ObjectType.AgentT, GridLocation(r, c, GridLayer.Agent_Layer))
        MettaObject.init_mo(1)
        this.energy = 100
        this.orientation = 0
        this.inventory.resize(InventoryItem.InventoryCount)

    inline void update_inventory(InventoryItem item, short amount):
        this.inventory[<InventoryItem>item] += amount

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = this.hp
        obs[2] = this.energy
        obs[3] = this.orientation
        obs[4] = this.shield

        cdef unsigned short idx = 5
        cdef unsigned short i
        for i in range(InventoryItem.InventoryCount):
            obs[idx + i] = this.inventory[i]

    @staticmethod
    inline vector[string] feature_names():
        return [
            "agent", "agent:hp", "agent:energy", "agent:orientation",
            "agent:shield"

        ] + [
            "agent:inv:" + n for n in InventoryItemNames]

cdef cppclass Wall(MettaObject):
    inline Wall(GridCoord r, GridCoord c):
        GridObject.init(ObjectType.WallT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(1)

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp

    @staticmethod
    inline vector[string] feature_names():
        return ["wall", "wall:hp"]

cdef cppclass Generator(Usable):
    unsigned int r1

    inline Generator(GridCoord r, GridCoord c):
        GridObject.init(ObjectType.GeneratorT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(1)
        Usable.init_usable(0, 10)
        this.r1 = 10

    inline char usable(const Agent *actor):
        return Usable.usable(actor) and this.r1 > 0

    inline void obs(int[:] obs):
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

    inline Converter(GridCoord r, GridCoord c):
        GridObject.init(ObjectType.ConverterT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(1)
        Usable.init_usable(0, 3)
        this.input_resource = InventoryItem.r1
        this.output_resource = InventoryItem.r2
        this.output_energy = 10

    inline char usable(const Agent *actor):
        return Usable.usable(actor) and actor.inventory[this.input_resource] > 0

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

cdef cppclass Altar(Usable):
    inline Altar(GridCoord r, GridCoord c):
        GridObject.init(ObjectType.AltarT, GridLocation(r, c, GridLayer.Object_Layer))
        MettaObject.init_mo(1)
        Usable.init_usable(10, 5)

    inline void obs(int[:] obs):
        obs[0] = 1
        obs[1] = hp
        obs[2] = ready

    @staticmethod
    inline vector[string] feature_names():
        return ["altar", "altar:hp", "altar:ready"]


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

