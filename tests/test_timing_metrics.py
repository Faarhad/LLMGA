import unittest

from desim.event import Event
from desim.models import PowerProfile, ResourceCapacity, Task, VirtualMachine
from desim.simulation import Simulation
from desim.timing_metrics import TimingMetricsCollector
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


def make_task(task_id: str, arrival: float = 0.0, workload: float = 1000.0) -> Task:
    return Task(
        task_id=task_id,
        workload_mi=workload,
        arrival_time=arrival,
        deadline=100,
        cpu_demand_mips=100,
        memory_demand_mb=128,
        io_size_mb=10,
    )


class TestTimingMetricsCollector(unittest.TestCase):
    def test_event_only_task_timing_metrics(self) -> None:
        sim = Simulation()
        vm = make_vm("vm1", availability=3.0)
        vm_exec = VirtualMachineExecutionManager.from_virtual_machines([vm])
        timing = TimingMetricsCollector.from_virtual_machines([vm])

        vm_exec.register(sim)
        timing.register(sim)

        t1 = make_task("t1", arrival=0.0, workload=1000.0)  # duration 1s
        t2 = make_task("t2", arrival=0.0, workload=1000.0)  # duration 1s

        sim.schedule(Event(time=0.0, name="vm.enqueue", payload={"vm_id": "vm1", "task": t1, "duration": 1.0}))
        sim.schedule(Event(time=0.0, name="vm.enqueue", payload={"vm_id": "vm1", "task": t2, "duration": 1.0}))
        sim.run()

        snapshot = timing.snapshot(sim.state)
        m1 = snapshot.task_metrics["t1"]
        m2 = snapshot.task_metrics["t2"]

        self.assertEqual(m1.waiting_time, 3.0)
        self.assertEqual(m1.execution_time, 1.0)
        self.assertEqual(m1.completion_time, 4.0)
        self.assertEqual(m1.response_time, 4.0)
        self.assertEqual(m1.queue_delay, 3.0)
        self.assertEqual(m1.carry_over, 3.0)

        self.assertEqual(m2.waiting_time, 4.0)
        self.assertEqual(m2.execution_time, 1.0)
        self.assertEqual(m2.completion_time, 5.0)
        self.assertEqual(m2.response_time, 5.0)
        self.assertEqual(m2.queue_delay, 4.0)
        self.assertEqual(m2.carry_over, 3.0)

        self.assertEqual(snapshot.makespan, 5.0)

        vm_metrics = snapshot.vm_metrics["vm1"]
        self.assertEqual(vm_metrics.initial_availability, 3.0)
        self.assertEqual(vm_metrics.busy_intervals, [(3.0, 4.0), (4.0, 5.0)])
        self.assertEqual(vm_metrics.idle_intervals, [])
        self.assertTrue(len(vm_metrics.availability_updates) >= 3)

    def test_queue_delay_differs_from_waiting_when_arrival_after_enqueue_time(self) -> None:
        sim = Simulation()
        vm = make_vm("vm1", availability=0.0)
        vm_exec = VirtualMachineExecutionManager.from_virtual_machines([vm])
        timing = TimingMetricsCollector.from_virtual_machines([vm])

        vm_exec.register(sim)
        timing.register(sim)

        t1 = make_task("t1", arrival=2.0, workload=1000.0)
        sim.schedule(Event(time=2.0, name="vm.enqueue", payload={"vm_id": "vm1", "task": t1, "duration": 1.0}))
        sim.run()

        snapshot = timing.snapshot(sim.state)
        m1 = snapshot.task_metrics["t1"]
        self.assertEqual(m1.waiting_time, 0.0)
        self.assertEqual(m1.queue_delay, 0.0)
        self.assertEqual(m1.carry_over, 0.0)


if __name__ == "__main__":
    unittest.main()
