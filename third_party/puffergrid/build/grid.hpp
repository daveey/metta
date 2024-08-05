#ifndef GRID_HPP
#define GRID_HPP

#include "grid_object.hpp"
#include <vector>
#include <algorithm>

using namespace std;
typedef vector<vector<vector<GridObjectId> > > GridType;

class Grid {
    public:
        unsigned int width;
        unsigned int height;
        vector<Layer> layer_for_type_id;
        Layer num_layers;

        GridType grid;
        vector<GridObjectBase*> objects;

        inline Grid(unsigned int width, unsigned int height, vector<Layer> layer_for_type_id)
            : width(width), height(height), layer_for_type_id(layer_for_type_id) {

                num_layers = *max_element(layer_for_type_id.begin(), layer_for_type_id.end()) + 1;
                grid.resize(height, vector<vector<GridObjectId> >(
                    width, vector<GridObjectId>(this->num_layers, 0)));

                // 0 is reserved for empty space
                objects.push_back(nullptr);
        }

        template <typename T>
        inline T* create_object(TypeId type_id, const GridLocation &loc) {
            if (this->grid[loc.r][loc.c][loc.layer] != 0) {
                return nullptr;
            }

            T* obj = new T(type_id);
            obj->location = loc;
            obj->id = this->objects.size();
            this->objects.push_back(obj);
            this->grid[loc.r][loc.c][loc.layer] = obj->id;
            return obj;
        }

        template <typename T>
        inline T* create_object(TypeId type_id, unsigned int row, unsigned int col) {
            if (type_id >= layer_for_type_id.size()) {
                return nullptr;
            }
            GridLocation loc = {row, col, layer_for_type_id[type_id]};
            return create_object<T>(type_id, loc);
        }

        inline char move_object(GridObjectId id, const GridLocation &loc) {
            if (grid[loc.r][loc.c][loc.layer] != 0) {
                return false;
            }

            GridObjectBase* obj = objects[id];
            grid[loc.r][loc.c][loc.layer] = id;
            grid[obj->location.r][obj->location.c][obj->location.layer] = 0;
            obj->location = loc;
            return true;
        }

        inline GridObjectBase* object(GridObjectId obj_id) {
            return objects[obj_id];
        }

        template <typename T>
        inline T* object(GridObjectId obj_id) {
            return dynamic_cast<T*>(objects[obj_id]);
        }

        inline GridObjectBase* object_at(const GridLocation &loc) {
            return objects[grid[loc.r][loc.c][loc.layer]];
        }

        template <typename T>
        inline T* object_at(const GridLocation &loc) {
            return dynamic_cast<T*>(objects[grid[loc.r][loc.c][loc.layer]]);
        }

        inline const GridLocation location(GridObjectId id) {
            return objects[id]->location;
        }

        inline const GridLocation relative_location(const GridLocation &loc, Orientation orientation) {
            GridLocation new_loc = loc;
            switch (orientation) {
                case Up:
                    if (loc.r > 0) new_loc.r = loc.r - 1;
                    break;
                case Down:
                    new_loc.r = loc.r + 1;
                    break;
                case Left:
                    if (loc.c > 0) new_loc.c = loc.c - 1;
                    break;
                case Right:
                    new_loc.c = loc.c + 1;
                    break;
                default:
                    printf("_relativelocation: Invalid orientation: %d\n", orientation);
                    break;
            }
            return new_loc;
        }

        inline char is_empty(unsigned int row, unsigned int col) {
            for (int layer = 0; layer < num_layers; ++layer) {
                if (grid[row][col][layer] != 0) {
                    return 0;
                }
            }
            return 1;
        }
};

#endif // GRID_HPP
