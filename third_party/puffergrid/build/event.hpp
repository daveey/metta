#ifndef EVENT_H
#define EVENT_H

#include <queue>
using namespace std;

struct Event {
    unsigned int timestamp;
    unsigned short event_id;
    unsigned short object_id;
    unsigned int arg;

    bool operator<(const Event& other) const {
        return timestamp > other.timestamp;
    }
};

class EventManager {
    public:
        priority_queue<Event> event_queue;

        void schedule_event(
            unsigned int delay,
            unsigned short event_id,
            unsigned short object_id,
            int arg);

        void process_events(unsigned int timestamp);
};

#endif // EVENT_H
