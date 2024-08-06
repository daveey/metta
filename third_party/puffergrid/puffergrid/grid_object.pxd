from typing import Type
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "grid_object.hpp":
    ctypedef unsigned short Layer
    ctypedef unsigned short TypeId

    ctypedef unsigned int GridCoord
    cdef struct GridLocation:
        GridCoord r
        GridCoord c
        Layer layer

    ctypedef enum Orientation:
        Up = 0
        Down = 1
        Left = 2
        Right = 3

    ctypedef unsigned int GridObjectId

    cdef cppclass GridObjectBase:
        GridObjectId id
        GridLocation location
        TypeId _type_id

        GridObjectBase(TypeId type_id)
        void __dealloc__()

    cdef cppclass GridObject[T] (GridObjectBase):
        T* props

        GridObject(TypeId type_id)
        void __dealloc__()

        @staticmethod
        GridObject[T]* create(TypeId type_id)

        void __dealloc__()
