from libc.stdio cimport printf

import numpy as np
from env.puffergrid.action cimport ActionArg, ActionHandler
from env.puffergrid.grid_object cimport Layer, GridLocation

cdef class GridEnv:
    def __init__(
            self,
            unsigned int map_width,
            unsigned int map_height,
            vector[Layer] layer_for_type_id,
            observation_encoder,
        ):

        self._grid = new Grid(map_width, map_height, layer_for_type_id)
        self._event_manager = new EventManager()
        self._obs_encoder = observation_encoder
        self._current_timestep = 0

    cpdef void reset(
        self,
        GridObjectId[:] actor_ids,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:,:] obs):
        pass

    cpdef void step(
        self,
        GridObjectId[:] actor_ids,
        unsigned int[:,:] actions,
        float[:] rewards,
        char[:] dones,
        unsigned short obs_width,
        unsigned short obs_height,
        int[:,:,:,:] obs):

        cdef:
            unsigned int idx
            float reward
            char done
            unsigned int num_actors = actor_ids.shape[0]
            cdef short action
            cdef ActionArg arg

        self._current_timestep += 1
        self._process_events()

        for idx in range(num_actors):
            reward = 0
            done = 0
            action = actions[idx][0]
            arg = actions[idx][1]
        #   self._action_handlers[action].handle_action(
        #       self._objects[actor_ids[idx]],
        #       arg,
        #       &reward,
        #       &done
        #   )

            rewards[idx] = reward
            dones[idx] = done

        # self.compute_observations(actor_ids, obs_width, obs_height, obs)



    ###############################
    # Python API
    ###############################

    cpdef grid(self):
        return []

    cpdef unsigned int num_features(self):
        return self._num_features

    cpdef unsigned int current_timestep(self):
        return self._current_timestep

    cpdef unsigned int map_width(self):
        return self._grid.width

    cpdef unsigned int map_height(self):
        return self._grid.height

    cpdef list[str] grid_features(self):
        return self._grid_features

