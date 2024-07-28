        self._observations = np.zeros((
            num_agents,
            obs_width,
            obs_height
        ), dtype=self._grid_dtype)

        self._actions = np.zeros((num_agents, 2), dtype=np.uint8)

        self._rewards = np.zeros(num_agents, dtype=np.float32)
        self._terminals = np.zeros(num_agents, dtype=np.uint8)
        self._truncations = np.zeros(num_agents, dtype=np.uint8)
        self._masks = np.ones(num_agents, dtype=np.uint8)



    def get_object(self, int object_id):
        return self._objects[object_id]

cdef class GridObjectType:
    def __init__(self, name, grid_layer, properties, is_observer=False):
        self._name = name
        self._id = -1
        self._grid_layer = grid_layer
        self._properties = properties
        self._is_observer = is_observer

    def id(self):
        return self._id

    def set_id(self, int _id):
        self._id = _id

    def name(self):
        return self._name

    def grid_layer(self):
        return self._grid_layer

    def properties(self):
        return self._properties

    def is_observer(self):
        return self._is_observer

cdef class GridObject():

    def __init__(
        self, int id, props_dtype, props,
        int r = -1, int c = -1):
        self._id = id
        self._props_dtype = props_dtype
        self._props = np.array(props, dtype=self._props_dtype)
        self._location = GridLocation(r, c)

    def id(self):
        return self._id

    def props(self):
        return self._props

    def location(self):
        return self._location

    def layer(self):
        return -1

    cdef char[:] _obj_data_as_np(self, GridObject obj):
        data_size = self._object_types[obj.type_id].object_size
        cdef char[:] array = cnp.PyArray_SimpleNewFromData(
            1, [data_size], cnp.NPY_BYTE, obj.data)
        return array
