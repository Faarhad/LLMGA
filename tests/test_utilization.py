import unittest

from desim.event import Event
from desim.models import PowerProfile, ResourceCapacity, Task, VirtualMachine
from desim.simulation import Simulation
from desim.utilization import UtilizationTracker
from desim.vm_execution import VirtualMachineExecutionManager


def make_vm(vm_id: str = "vm1", availability: float = 0.0) -> VirtualMachine:
    return VirtualMachine(
        vm_id=vm_id,
        vm_type="small",
        host_machine_id="pm1",
        capacity=ResourceCapacity(cpu_mips=1000, memory_mb=2048, bandwidth_mbps=100),
        power=PowerProfile(idle_watts=20, max_watts=60),
        vcpu_count=2,
        availability_time=availability,
    )


def make_task(task_id: str, arrival: float, cpu_demand: float, workload: float) -> Task:
    return Task(
        task_id=task_id,
        workload_mi=workload,
        arrival_time=arrival,
        deadline=100,
        cpu_demand_mips=cpu_demand,
        memory_demand_mb=128,
        io_size_mb=10,
    )


class TestUtilizationTracker(unittest.TestCase):
    def test_intervals_change_only_on_start_finish_events(self) -> None:
        sim = Simulation()
        vm = make_vm("vm1")
        vm_exec = VirtualMachineExecutionManager.from_virtual_machines([vm])
        tracker = UtilizationTracker.from_virtual_machines([vm])

        vm_exec.register(sim)
        tracker.register(sim)

        t1 = make_task(task_id="t1", arrival=1.0, cpu_demand=400.0, workload=1000.0)  # duration 1s
        t2 = make_task(task_id="t2", arrival=3.0, cpu_demand=200.0, workload=1000.0)  # duration 1s

        sim.schedule(Event(time=1.0, name="vm.enqueue", payload={"vm_id": "vm1", "task": t1, "duration": 1.0}))
        sim.schedule(Event(time=3.0, name="vm.enqueue", payload={"vm_id": "vm1", "task": t2, "duration": 1.0}))

        sim.run(until=5.0)
        tracker.finalize(end_time=5.0)

        trace = tracker.snapshot().traces["vm1"]

        # Expected changes only at 1.0(start t1), 2.0(finish t1), 3.0(start t2), 4.0(finish t2), finalized at 5.0.
        expected = [
            (0.0, 1.0, 0.0),
            (1.0, 2.0, 0.4),
            (2.0, 3.0, 0.0),
            (3.0, 4.0, 0.2),
            (4.0, 5.0, 0.0),
        ]
        got = [(i.start_time, i.end_time, i.utilization) for i in trace.intervals]
        self.assertEqual(got, expected)

    def test_no_periodic_sampling_when_no_events(self) -> None:
        sim = Simulation()
        vm = make_vm("vm1")
        tracker = UtilizationTracker.from_virtual_machines([vm])
        tracker.register(sim)

        sim.run(until=10.0)

        trace_before_finalize = tracker.snapshot().traces["vm1"]
        self.assertEqual(trace_before_finalize.intervals, [])

        tracker.finalize(end_time=10.0)
        trace_after_finalize = tracker.snapshot().traces["vm1"]
        self.assertEqual(len(trace_after_finalize.intervals), 1)
        self.assertEqual(trace_after_finalize.intervals[0].start_time, 0.0)
        self.assertEqual(trace_after_finalize.intervals[0].end_time, 10.0)
        self.assertEqual(trace_after_finalize.intervals[0].utilization, 0.0)


if __name__ == "__main__":
    unittest.main()
