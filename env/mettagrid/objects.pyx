# distutils: language=c++

from libc.stdio cimport printf
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport cython

from puffergrid.grid_object cimport GridObjectBase, GridObjectId
from puffergrid.event cimport EventHandler, EventArg

cdef class MettaObservationEncoder(ObservationEncoder):
    def __init__(self) -> None:
        self._offsets.resize(ObjectType.Count)

        features = []
        self._offsets[ObjectType.AgentT] = 0
        features.extend(AgentProps.feature_names())

        self._offsets[ObjectType.WallT] = len(features)
        features.extend(WallProps.feature_names())

        self._offsets[ObjectType.GeneratorT] = len(features)
        features.extend(GeneratorProps.feature_names())

        self._offsets[ObjectType.ConverterT] = len(features)
        features.extend(ConverterProps.feature_names())

        self._offsets[ObjectType.AltarT] = len(features)
        features.extend(AltarProps.feature_names())

        self._feature_names = features

    cdef encode(self, GridObjectBase *obj, int[:] obs):
        cdef vector[int] object_obs

        if obj._type_id == ObjectType.AgentT:
            (<Agent*>obj).props.obs(obs[self._offsets[ObjectType.AgentT]:])
        elif obj._type_id == ObjectType.WallT:
            (<Wall*>obj).props.obs(obs[self._offsets[ObjectType.WallT]:])
        elif obj._type_id == ObjectType.GeneratorT:
            (<Generator*>obj).props.obs(obs[self._offsets[ObjectType.GeneratorT]:])
        elif obj._type_id == ObjectType.ConverterT:
            (<Converter*>obj).props.obs(obs[self._offsets[ObjectType.ConverterT]:])
        elif obj._type_id == ObjectType.AltarT:
            (<Altar*>obj).props.obs(obs[self._offsets[ObjectType.AltarT]:])
        else:
            printf("Encoding object of unknown type: %d\n", obj._type_id)

    cdef vector[string] feature_names(self):
        return self._feature_names
