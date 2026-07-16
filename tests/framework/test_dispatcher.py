import unittest

from desim.framework.dispatcher import EventDispatcher
from desim.framework.event import Event
from desim.framework.state import SimulationState


class TestEventDispatcher(unittest.TestCase):
    def test_registered_handler_runs(self) -> None:
        dispatcher = EventDispatcher()
        state = SimulationState()

        def on_ping(s, e):
            s.set("pong", e.payload["value"])

        dispatcher.register("ping", on_ping)
        dispatcher.dispatch(state, Event(time=0.0, name="ping", payload={"value": 7}))

        self.assertEqual(state.get("pong"), 7)

    def test_event_local_handler_runs(self) -> None:
        dispatcher = EventDispatcher()
        state = SimulationState()

        def local_handler(s, e):
            s.set("local", e.name)

        event = Event(time=0.0, name="x", handler=local_handler)
        dispatcher.dispatch(state, event)

        self.assertEqual(state.get("local"), "x")


if __name__ == "__main__":
    unittest.main()

