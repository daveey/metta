
from libc.stdio cimport printf

from puffergrid.grid_object cimport GridLocation, GridObjectId, Orientation, GridObject
from puffergrid.action cimport ActionHandler, ActionArg
from env.mettagrid.objects cimport MettaObject, ObjectType, Usable, Altar, Agent, Events, GridLayer
from env.mettagrid.objects cimport Generator, Converter, InventoryItem, ObjectTypeNames, InventoryItemNames
from env.mettagrid.actions.actions cimport MettaActionHandler

cdef class Shield(MettaActionHandler):
    def __init__(self):
        MettaActionHandler.__init__(self, "shield")

    cdef char _handle_action(
        self,
        unsigned int actor_id,
        Agent * actor,
        ActionArg arg):

        if actor.shield:
            actor.shield = True
        else:
            actor.shield = False

