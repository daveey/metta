from libcpp.queue cimport priority_queue

cdef extern from "event.hpp":
    cdef struct Event:
        unsigned short event_id
        unsigned short object_id
        unsigned int arg

    cdef cppclass EventManager:
        priority_queue[Event] event_queue
        unsigned int current_timestep

        void schedule_event(
            unsigned int delay,
            unsigned short event_id,
            unsigned short object_id,
            int arg)

        void process_events(unsigned int current_timestep)
