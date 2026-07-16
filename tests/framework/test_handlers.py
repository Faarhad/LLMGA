import unittest

from desim.framework.dispatcher import EventDispatcher
from desim.framework.event import Event
from desim.framework.handlers import CallbackEventHandler
from desim.framework.state import SimulationState


class TestCallbackEventHandler(unittest.TestCase):
    def test_register_and_handle(self) -> None:
        dispatcher = EventDispatcher()
        state = SimulationState()

        def cb(s, e):
            s.set("handled", e.name)

        h = CallbackEventHandler(event_name="ping", callback=cb)
        h.register(dispatcher)

        dispatcher.dispatch(state, Event(time=0, name="ping"))
        self.assertEqual(state.get("handled"), "ping")

    def test_empty_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CallbackEventHandler(event_name="", callback=lambda s, e: None)


if __name__ == "__main__":
    unittest.main()

