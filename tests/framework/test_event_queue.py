import unittest

from desim.framework.event import Event
from desim.framework.queue import EventQueue


class TestEventQueue(unittest.TestCase):
    def test_event_time_order(self) -> None:
        q = EventQueue()
        q.schedule(Event(time=3.0, name="c"))
        q.schedule(Event(time=1.0, name="a"))
        q.schedule(Event(time=2.0, name="b"))

        self.assertEqual(q.pop_next().name, "a")
        self.assertEqual(q.pop_next().name, "b")
        self.assertEqual(q.pop_next().name, "c")

    def test_next_time(self) -> None:
        q = EventQueue()
        self.assertIsNone(q.next_time())
        q.schedule(Event(time=4.0, name="x"))
        self.assertEqual(q.next_time(), 4.0)


if __name__ == "__main__":
    unittest.main()

