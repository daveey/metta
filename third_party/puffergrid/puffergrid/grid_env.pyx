from libc.stdio cimport printf

cimport numpy as cnp
import numpy as np
from puffergrid.action cimport ActionArg, ActionHandler
from puffergrid.grid_object cimport Layer, GridLocation
from puffergrid.observation_encoder cimport ObservationEncoder
from puffergrid.grid_object cimport GridObjectBase, GridObjectId
from puffergrid.event cimport EventManager
from puffergrid.grid cimport Grid
from libcpp.vector cimport vector

cdef class GridEnv:
    def __init__(
            self,
            unsigned int map_width,
            unsigned int map_height,
            vector[Layer] layer_for_type_id,
            unsigned short obs_width,
            unsigned short obs_height,
            ObservationEncoder observation_encoder,
            list[ActionHandler] action_handlers
        ):

        self._grid = new Grid(map_width, map_height, layer_for_type_id)
        self._event_manager = new EventManager()

        self.obs_width = obs_width
        self.obs_height = obs_height

        self._obs_encoder = observation_encoder
        self._action_handlers = action_handlers
        for handler in action_handlers:
            (<ActionHandler>handler).init(self)

        self._current_timestep = 0

    cdef void add_agent(self, GridObjectBase* agent):
        self._agents.push_back(agent)

    cdef void _compute_observation(
        self,
        GridObjectBase* observer,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation):

        cdef:
            GridLocation observer_loc = observer.location
            GridLocation object_loc
            GridObjectBase *obj
            unsigned short obs_width_r = obs_width >> 1
            unsigned short obs_height_r = obs_height >> 1
            cdef unsigned int obs_r, obs_c
            cdef int[:] agent_ob

        for object_loc.r in range(observer_loc.r - obs_height_r, observer_loc.r + obs_height_r + 1):
            if object_loc.r < 0 or object_loc.r >= self._grid.height:
                continue
            for object_loc.c in range(observer_loc.c - obs_width_r, observer_loc.c + obs_width_r + 1):
                if object_loc.c < 0 or object_loc.c >= self._grid.width:
                    continue
                for object_loc.layer in range(self._grid.num_layers):
                    obj = self._grid.object_at(object_loc)
                    if obj == NULL:
                        continue

                    obs_r = object_loc.r - (observer_loc.r - obs_height_r)
                    obs_c = object_loc.c - (observer_loc.c - obs_width_r)
                    agent_ob = observation[:, obs_r, obs_c]
                    self._obs_encoder.encode(obj, agent_ob)

    cdef void _compute_observations(self):
        for idx in range(self._agents.size()):
            self._compute_observation(self._agents[idx], self.obs_width, self.obs_height, self._observations[idx])

    cdef void _step(self, unsigned int[:,:] actions):
        cdef:
            unsigned int idx
            short action
            ActionArg arg
            GridObjectBase *agent
            ActionHandler handler

        self._current_timestep += 1
        # self._process_events()

        for idx in range(self._agents.size()):
            self._rewards[idx] = 0
            self._dones[idx] = 0
            action = actions[idx][0]
            arg = actions[idx][1]
            agent = self._agents[idx]
            handler = <ActionHandler>self._action_handlers[action]
            handler.handle_action(idx, agent.id, arg)
        self._compute_observations()

    ###############################
    # Python API
    ###############################
    cpdef void reset(self):
        self._compute_observations()

    cpdef void step(self, unsigned int[:,:] actions):
        self._step(actions)

    cpdef void set_buffers(
        self,
        cnp.ndarray[int, ndim=4] observations,
        cnp.ndarray[char, ndim=1] dones,
        cnp.ndarray[float, ndim=1] rewards):

        self._observations = observations
        self._dones = dones
        self._rewards = rewards

    cpdef grid(self):
        return []

    cpdef unsigned int num_actions(self):
        return len(self._action_handlers)

    cpdef unsigned int current_timestep(self):
        return self._current_timestep

    cpdef unsigned int map_width(self):
        return self._grid.width

    cpdef unsigned int map_height(self):
        return self._grid.height

    cpdef list[str] grid_features(self):
        return self._obs_encoder.feature_names()

    cpdef compute_observation(
        self,
        GridObjectId observer_id,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:] observation):
        self._compute_observation(self._grid.object(observer_id), obs_width, obs_height, observation)
