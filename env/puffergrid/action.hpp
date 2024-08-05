#ifndef ACTION_HPP
#define ACTION_HPP

#include "grid.hpp"
#include "grid_object.hpp"
#include "event.hpp"

typedef unsigned int ActionArg;

class ActionHandler {
public:
    Grid* _grid;

    inline ActionHandler() : _grid(nullptr) {}

    inline void init(Grid* grid, EventManager* event_manager) {
        this->_grid = grid;
    }

    virtual char handle_action(
        GridObjectBase* actor,
        ActionArg arg,
        float* reward,
        char* done) = 0;


};

#endif // ACTION_HPP
