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


    cdef unsigned int create_object(
        self, TypeId type_id, GridLocation location):
        pass

        if location.r >= 0 and location.c >= 0:
            if self._grid[location.r, location.c, location.layer] != 0:
                return 0

        object = GridObject(
            object_id,
            type_id,
            GridLocation(location.r, location.c, location.layer),
            <void *> data.data
        )
        self._objects.push_back(object)

        if location.r >= 0 and location.c >= 0:
            self._grid[location.r, location.c, location.layer] = object_id

        return object_id

