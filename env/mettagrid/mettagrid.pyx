# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
# distutils: language = c++
import cython
from libc.stdio cimport printf

import numpy as np
cimport numpy as cnp

from env.puffergrid.grid_object cimport GridLocation, GridObject
from env.puffergrid.grid cimport PufferGrid, Event

cdef unsigned int GridLayer_Agent = 0
cdef unsigned int GridLayer_Object = 1

cdef struct Agent:
    unsigned int id
    unsigned int hp
    unsigned int energy
    unsigned int orientation

cdef struct Wall:
    unsigned int hp

cdef struct Tree:
    unsigned int has_food
    unsigned int cooldown

cdef unsigned short ACTION_ROTATE = 0
cdef unsigned short ACTION_MOVE = 1
cdef unsigned short ACTION_EAT = 2
cdef unsigned short ACTION_COUNT = 3

cdef unsigned short EVENT_RESET_TREE = 0

cdef class MettaGrid(PufferGrid):
    cdef:
        int AgentType
        int WallType
        int TreeType

    cdef encode_object(
        self,
        int object_type,
        int object_id,
        cnp.ndarray destination):
        pass

    def __init__(self, *args, **kwargs):

        cdef Agent agent
        agent_dtype = np.asarray(<Agent[:1]>&agent).dtype
        cdef Wall wall
        wall_dtype = np.asarray(<Wall[:1]>&wall).dtype
        cdef Tree tree
        tree_dtype = np.asarray(<Tree[:1]>&tree).dtype

        super().__init__(
            {
                "Agent": (Agent(), agent_dtype, GridLayer_Agent),
                "Wall": (Wall(), wall_dtype, GridLayer_Object),
                "Tree": (Tree(), tree_dtype, GridLayer_Object),
            },
            *args,
            num_layers=2,
            **kwargs
        )
        self.AgentType = self._type_ids["Agent"]
        self.WallType = self._type_ids["Wall"]
        self.TreeType = self._type_ids["Tree"]


    ################
    # Python Interface
    ################

    ################
    # Actions
    ################
    @cython.cdivision(True)
    cdef void handle_action(
        self, unsigned int actor_id,
        unsigned short action_id,
        unsigned short action_arg,
        float *reward, char *done):

        action_id = action_id % ACTION_COUNT

        if action_id == ACTION_MOVE:
            self._agent_move(actor_id, action_arg)
        elif action_id == ACTION_ROTATE:
            self._agent_rotate(actor_id, action_arg)
        elif action_id == ACTION_EAT:
            self._agent_eat(actor_id, reward)

        else:
            printf("Unhandled Action: %d %d %d\n", actor_id, action_id, action_arg)

    cdef void _agent_rotate(self, unsigned int agent_id, unsigned int orientation):
        self._agent(agent_id).orientation = (orientation % 4)
        cdef GridLocation loc = self._target_loc(agent_id)
        # printf("action_rotate: %d -> %d %d\n", orientation, loc.r, loc.c)

    cdef void _agent_move(self, unsigned int agent_id, unsigned int direction):
        # direction can be forward (0) or backward (1)
        cdef Agent * agent = <Agent *>self._agent(agent_id)
        cdef unsigned short orientation = (agent.orientation + 2*(direction % 2)) % 4
        cdef GridLocation loc = self.location(agent_id)
        loc = self._relative_location(loc, orientation)
        # printf("action_move: %d -> %d %d\n", direction, loc.r, loc.c)
        self.move_object(agent_id, loc.r, loc.c)

    cdef void _agent_eat(MettaGrid self, unsigned int agent_id, float *reward):
        cdef Agent * agent = self._agent(agent_id)
        cdef unsigned int target_id = self._target(agent_id, GridLayer_Object)
        cdef Tree *tree

        # printf("action_eat: %d\n", target_id)
        if target_id == 0:
            return

        target_obj = self._objects[target_id]
        if target_obj.type_id != self.TreeType:
            return

        tree = <Tree*>target_obj.data
        if tree.has_food:
            tree.has_food = 0
            agent.energy += 1
            reward += 1
            self._schedule_event(10, EVENT_RESET_TREE, target_id, 0)
            printf("Ate food\n")


    cdef void _reset_tree(self, unsigned int tree_id):
        cdef GridObject obj = self._objects[tree_id]
        if obj.type_id != self.TreeType:
            printf("Invalid tree id: %d\n", tree_id)
            return

        cdef Tree *tree = <Tree*>self._objects[tree_id].data
        tree.has_food = 1
        printf("Tree reset\n")

    ################
    # Events
    ################

    cdef void handle_event(self, Event event):
        printf("Event: %d %d %d %d\n", event.timestamp, event.event_id, event.object_id, event.arg)
        if event.event_id == EVENT_RESET_TREE:
            self._reset_tree(event.object_id)

    ################
    # Helpers
    ################

    cdef Agent * _agent(self, unsigned int agent_id):
        cdef GridObject obj = self._objects[agent_id]
        return <Agent*>obj.data

    cdef GridLocation _target_loc(self, unsigned int agent_id):
        return self._relative_location(
            self._objects[agent_id].location,
            self._agent(agent_id).orientation
        )

    cdef unsigned int _target(self, unsigned int agent_id, unsigned short layer):
        cdef GridLocation loc = self._target_loc(agent_id)
        return self._grid[loc.r, loc.c, layer]


