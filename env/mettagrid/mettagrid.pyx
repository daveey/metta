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
from env.puffergrid.grid cimport PufferGrid, Event, Action
from env.puffergrid.stats_tracker cimport StatsTracker
from omegaconf import OmegaConf
from env.griddly.mettagrid.game_builder import MettaGridGameBuilder

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

cdef enum Actions:
    Rotate = 0
    Move = 1
    Eat = 2
    Count = 3

cdef enum Events:
    TreeReset = 0

cdef struct TypeIds:
    unsigned int Agent
    unsigned int Wall
    unsigned int Tree

cdef class MettaGrid(PufferGrid):
    cdef:
        TypeIds _type
        StatsTracker _stats
        object _cfg

    def __init__(self, cfg: OmegaConf, map: np.ndarray):
        self._cfg = cfg

        cdef Agent agent
        cdef Wall wall
        cdef Tree tree

        super().__init__(
            {
                "Agent": np.asarray(<Agent[:1]>&agent).dtype,
                "Wall": np.asarray(<Wall[:1]>&wall).dtype,
                "Tree": np.asarray(<Tree[:1]>&tree).dtype,
            },
            num_layers=2,
            map_width=map.shape[0],
            map_height=map.shape[1]
        )
        self._type.Agent = self._type_ids["Agent"]
        self._type.Wall = self._type_ids["Wall"]
        self._type.Tree = self._type_ids["Tree"]

        self._stats = StatsTracker(self._cfg.num_agents)

    ################
    # Python Interface
    ################
    cpdef unsigned int layer(self, unsigned int type_id):
        if type_id == self._type.Agent:
            return GridLayer_Agent
        return GridLayer_Object

    cpdef get_episode_stats(self):
        return self._stats.to_pydict()

    cpdef unsigned int num_actions(self):
        return Actions.Count

    ################
    # Actions
    ################
    @cython.cdivision(True)
    cdef void handle_action(
        self,
        const Action &action,
        float *reward, char *done):

        cdef char success = 0

        if action.id  == Actions.Move:
            success = self._agent_move(action, reward)
        elif action.id  == Actions.Rotate:
            success = self._agent_rotate(action, reward)
        elif action.id  == Actions.Eat:
            success = self._agent_eat(action, reward)

    cdef char _agent_rotate(self, const Action &action, float *reward):
        cdef unsigned short orientation = action.arg
        if orientation >= 4:
            return False

        self._stats.agent_incr(action.agent_idx, "action_rotate", 1)
        self._agent(action.actor_id).orientation = orientation
        return True

    cdef char _agent_move(self, const Action &action, float *reward):
        # direction can be forward (0) or backward (1)
        cdef unsigned short direction = action.arg

        if direction >= 2:
            return False

        cdef Agent * agent = <Agent *>self._agent(action.actor_id)
        cdef unsigned short orientation = (agent.orientation + 2*(direction)) % 4
        cdef GridLocation old_loc = self.location(action.actor_id)
        cdef GridLocation new_loc = self._relative_location(old_loc, orientation)

        if self._grid[new_loc.r, new_loc.c, GridLayer_Object] != 0 \
            or self._grid[new_loc.r, new_loc.c, GridLayer_Agent] != 0:
            return False
        cdef char s = self.move_object(action.actor_id, new_loc.r, new_loc.c)
        if s:
            self._stats.agent_incr(action.agent_idx, "action_move", 1)

    cdef char _agent_eat(MettaGrid self, const Action &action, float *reward):
        cdef Agent * agent = self._agent(action.actor_id)
        cdef unsigned int target_id = self._target(action.actor_id, GridLayer_Object)
        cdef Tree *tree

        # printf("action_eat: %d\n", target_id)
        if target_id == 0:
            return False

        target_obj = self._objects[target_id]
        if target_obj.type_id != self._type.Tree:
            return False

        tree = <Tree*>target_obj.data
        if tree.has_food == 0:
            return False

        tree.has_food = 0
        agent.energy += 1
        self._stats.agent_incr(action.agent_idx, "action_eat", 1)
        self._stats.agent_incr(action.agent_idx, "energy_gained", 1)
        self._stats.agent_incr(action.agent_idx, "fruit_eaten", 1)
        reward[0] += 1
        self._schedule_event(100, Events.TreeReset, target_id, 0)
        return True



    ################
    # Events
    ################

    cdef void handle_event(self, Event event):
        if event.event_id == Events.TreeReset:
            self._reset_tree(event.object_id)

    cdef void _reset_tree(self, unsigned int tree_id):
        cdef GridObject obj = self._objects[tree_id]
        if obj.type_id != self._type.Tree:
            printf("Invalid tree id: %d\n", tree_id)
            return

        cdef Tree *tree = <Tree*>self._objects[tree_id].data
        tree.has_food = 1
        self._stats.game_incr("fruit_spawned", 1)

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


