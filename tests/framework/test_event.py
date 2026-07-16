import unittest

from desim.framework.event import Event


class TestEvent(unittest.TestCase):
    def test_valid_event(self) -> None:
        event = Event(time=1.0, name="tick", payload={"x": 1})
        self.assertEqual(event.time, 1.0)
        self.assertEqual(event.name, "tick")
        self.assertEqual(event.payload["x"], 1)

    def test_negative_time_raises(self) -> None:
        with self.assertRaises(ValueError):
            Event(time=-1.0, name="bad")

    def test_empty_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            Event(time=1.0, name="")


if __name__ == "__main__":
    unittest.main()

