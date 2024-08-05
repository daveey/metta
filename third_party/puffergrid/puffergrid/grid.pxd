from libcpp.vector cimport vector
from puffergrid.grid_object cimport Layer, TypeId, GridObjectId, GridObjectBase
from puffergrid.grid_object cimport GridLocation, Orientation

cdef extern from "grid.hpp":
    cdef cppclass Grid:
        unsigned int width
        unsigned int height
        Layer num_layers

        vector[vector[vector[int]]] grid
        vector[GridObjectBase*] objects

        Grid(unsigned int width, unsigned int height, vector[Layer] layer_for_type_id)

        char move_object(GridObjectId id, const GridLocation &loc)
        GridObjectBase* object(GridObjectId obj_id)
        GridObjectBase* object_at(const GridLocation &loc)
        const GridLocation location(GridObjectId id)
        const GridLocation relative_location(const GridLocation &loc, Orientation orientation)
        char is_empty(unsigned int r, unsigned int c)

        GridObjectBase* create_object(TypeId type_id, const GridLocation &loc)

        # Templated functions
        T* object[T](GridObjectId obj_id)
        T* object_at[T](const GridLocation &loc)
        T* create_object[T](TypeId type_id, const GridLocation &loc)
        T* create_object[T](TypeId type_id, unsigned int r, unsigned int c)
