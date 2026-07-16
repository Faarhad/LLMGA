import unittest

from desim.clock import SimulationClock


class TestSimulationClock(unittest.TestCase):
    def test_initial_time(self) -> None:
        clock = SimulationClock()
        self.assertEqual(clock.now, 0.0)

    def test_advance_forward(self) -> None:
        clock = SimulationClock()
        clock.advance_to(3.5)
        self.assertEqual(clock.now, 3.5)

    def test_advance_backward_raises(self) -> None:
        clock = SimulationClock(start_time=2.0)
        with self.assertRaises(ValueError):
            clock.advance_to(1.0)

    def test_reset(self) -> None:
        clock = SimulationClock(start_time=5.0)
        clock.reset()
        self.assertEqual(clock.now, 0.0)


if __name__ == "__main__":
    unittest.main()
