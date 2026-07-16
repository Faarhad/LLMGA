import unittest

from desim.framework.state import SimulationLifecycle, SimulationState


class TestSimulationState(unittest.TestCase):
    def test_default_state(self) -> None:
        s = SimulationState()
        self.assertEqual(s.lifecycle, SimulationLifecycle.CREATED)
        self.assertEqual(s.events_processed, 0)

    def test_set_get(self) -> None:
        s = SimulationState()
        s.set("k", 10)
        self.assertEqual(s.get("k"), 10)
        self.assertEqual(s.get("missing", 3), 3)

    def test_reset(self) -> None:
        s = SimulationState()
        s.lifecycle = SimulationLifecycle.RUNNING
        s.events_processed = 5
        s.set("x", 1)
        s.reset()

        self.assertEqual(s.lifecycle, SimulationLifecycle.CREATED)
        self.assertEqual(s.events_processed, 0)
        self.assertEqual(s.data, {})


if __name__ == "__main__":
    unittest.main()

