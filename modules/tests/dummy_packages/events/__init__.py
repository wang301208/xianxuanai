def create_event_bus(*args, **kwargs):
    class Bus:
        def publish(self, *args, **kwargs):
            pass

        def subscribe(self, *args, **kwargs):
            def cancel():
                pass
            return cancel

        def unsubscribe(self, *args, **kwargs):
            pass
    return Bus()

def set_event_bus(bus):
    pass
