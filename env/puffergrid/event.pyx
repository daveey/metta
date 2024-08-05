
cdef extern from "event.hpp":
    cdef struct Event:
        unsigned int timestamp
        unsigned short event_id
        unsigned short object_id
        unsigned int arg

    cdef cppclass EventManager:
        void schedule_event(
            unsigned int delay,
            unsigned short event_id,
            unsigned short object_id,
            int arg):
            cdef Event event = Event(
                this.current_timestep + delay,
                event_id,
                object_id,
                arg
            )
            # printf("Scheduling Event: %d %d %d\n", event.timestamp, event.event_id, event.object_id)
            this.event_queue.push(event)

        void process_events():
            cdef Event event
            while not this.event_queue.empty():
                event = this.event_queue.top()
                if event.timestamp > this.current_timestep:
                    break
                this.event_queue.pop()
