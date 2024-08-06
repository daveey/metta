#ifndef GRID_OBJECT_HPP
#define GRID_OBJECT_HPP

#include <vector>
#include <string>

using namespace std;

typedef unsigned short Layer;
typedef unsigned short TypeId;
typedef unsigned int GridCoord;

struct GridObjectType {
    TypeId type_id;
    string name;
    vector<string> property_names;
    Layer layer;
};

struct GridLocation {
    GridCoord r;
    GridCoord c;
    Layer layer;
};

enum Orientation {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3
};

typedef unsigned int GridObjectId;

class GridObjectBase {
    public:
        GridObjectId id;
        GridLocation location;
        const TypeId _type_id;

        GridObjectBase(TypeId type_id) : _type_id(type_id) {}
        virtual ~GridObjectBase() {}
};

template <typename T>
class GridObject : public GridObjectBase {
    public:
        T* props;

        static GridObject<T>* create(TypeId type_id) {
            T* props = new T();
            return new GridObject<T>(type_id, props);
        }

        ~GridObject() {
            delete props;
        }

        GridObject(TypeId type_id)
            : GridObjectBase(type_id) {
                props = new T();
            }
};

#endif // GRID_OBJECT_HPP
