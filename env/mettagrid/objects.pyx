from libcpp.string cimport string
from libcpp.vector cimport vector

cdef class MettaObservationEncoder(ObservationEncoder):
    cdef encode(self, GridObjectBase *obj, int[:] obs):
        cdef Agent *agent
        cdef Wall *wall
        cdef Tree *tree

        if obj._type_id == ObjectType.AgentT:
            agent = <Agent *>obj
            obs[0] = 1
            obs[1] = agent.props.hp
            obs[2] = agent.props.energy
            obs[3] = agent.props.orientation
        elif obj._type_id == ObjectType.WallT:
            wall = <Wall *>obj
            obs[4] = 1
            obs[5] = wall.props.hp
        elif obj._type_id == ObjectType.TreeT:
            tree = <Tree *>obj
            obs[6] = 1
            obs[7] = tree.props.hp
            obs[8] = tree.props.has_fruit
        else:
            printf("Encoding object of unknown type: %d\n", obj._type_id)

    cdef vector[string] feature_names(self):
        return [
            "agent", "agent:hp", "agent:energy", "agent:orientation",
            "wall", "wall:hp",
            "tree", "tree:hp", "tree:has_fruit"]

