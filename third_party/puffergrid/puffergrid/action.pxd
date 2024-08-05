from puffergrid.grid cimport Grid
from puffergrid.event cimport EventManager
from puffergrid.grid_object cimport GridObjectBase

cdef extern from "action.hpp":
    ctypedef unsigned int ActionArg

    cdef cppclass ActionHandler:
        Grid* _grid

        ActionHandler() except +

        void init(Grid* grid, EventManager* event_manager)

        char handle_action(
            GridObjectBase* actor,
            ActionArg arg,
            float* reward,
            char* done)

        void foo()
