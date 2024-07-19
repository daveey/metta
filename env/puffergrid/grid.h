#ifndef GRID_H
#define GRID_H

// Orientation definitions
unsigned short Orientation_Up = 0;
unsigned short Orientation_Down = 1;
unsigned short Orientation_Left = 2;
unsigned short Orientation_Right = 3;

struct Event {
    unsigned int timestamp;
    unsigned short event_id;
    unsigned short object_id;
    unsigned int arg;

    bool operator<(const Event& other) const {
        return timestamp > other.timestamp;
    }
};

#endif // GRID_H
