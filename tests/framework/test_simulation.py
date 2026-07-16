import unittest

from desim.framework.event import Event
from desim.framework.simulation import Simulation
from desim.framework.state import SimulationLifecycle


class TestSimulation(unittest.TestCase):
    def test_run_processes_events_in_time_order(self) -> None:
        sim = Simulation()
        output = []

        def handler(state, event):
            output.append((event.time, event.name))

        sim.schedule(Event(time=2.0, name="b", handler=handler))
        sim.schedule(Event(time=1.0, name="a", handler=handler))
        sim.schedule(Event(time=3.0, name="c", handler=handler))

        count = sim.run()

        self.assertEqual(count, 3)
        self.assertEqual(output, [(1.0, "a"), (2.0, "b"), (3.0, "c")])
        self.assertEqual(sim.clock.now, 3.0)
        self.assertEqual(sim.state.lifecycle, SimulationLifecycle.COMPLETED)

    def test_run_until_time(self) -> None:
        sim = Simulation()
        sim.schedule(Event(time=1.0, name="a"))
        sim.schedule(Event(time=5.0, name="b"))

        count = sim.run(until=2.0)

        self.assertEqual(count, 1)
        self.assertEqual(len(sim.event_queue), 1)
        self.assertEqual(sim.state.lifecycle, SimulationLifecycle.INITIALIZED)

    def test_run_max_events(self) -> None:
        sim = Simulation()
        sim.schedule(Event(time=1.0, name="a"))
        sim.schedule(Event(time=2.0, name="b"))

        count = sim.run(max_events=1)

        self.assertEqual(count, 1)
        self.assertEqual(len(sim.event_queue), 1)

    def test_stop_from_handler(self) -> None:
        sim = Simulation()

        def stop_handler(state, event):
            sim.stop()

        sim.schedule(Event(time=1.0, name="stop", handler=stop_handler))
        sim.schedule(Event(time=2.0, name="later"))

        count = sim.run()

        self.assertEqual(count, 1)
        self.assertEqual(sim.state.lifecycle, SimulationLifecycle.STOPPED)

    def test_reset(self) -> None:
        sim = Simulation()
        sim.schedule(Event(time=1.0, name="a"))
        sim.run(max_events=1)
        sim.reset()

        self.assertEqual(sim.clock.now, 0.0)
        self.assertEqual(len(sim.event_queue), 0)
        self.assertEqual(sim.state.lifecycle, SimulationLifecycle.CREATED)
        self.assertEqual(sim.state.events_processed, 0)


if __name__ == "__main__":
    unittest.main()

