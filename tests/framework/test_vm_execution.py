import unittest

from desim.framework.event import Event
from desim.framework.models import PowerProfile, ResourceCapacity, Task, VirtualMachine
from desim.framework.simulation import Simulation
from desim.framework.vm_execution import VirtualMachineExecutionManager, VirtualMachineExecutionState


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


def make_task(task_id: str) -> Task:
    return Task(
        task_id=task_id,
        workload_mi=1000,
        arrival_time=0,
        deadline=10,
        cpu_demand_mips=100,
        memory_demand_mb=128,
        io_size_mb=10,
    )


class TestVirtualMachineExecutionState(unittest.TestCase):
    def test_enqueue_dequeue_fifo(self) -> None:
        vm_state = VirtualMachineExecutionState(vm=make_vm())
        t1 = make_task("t1")
        t2 = make_task("t2")

        vm_state.enqueue(task=t1, duration=2.0, enqueued_at=0.0)
        vm_state.enqueue(task=t2, duration=1.0, enqueued_at=0.1)

        self.assertEqual(vm_state.queue_length(), 2)
        self.assertEqual(vm_state.dequeue().task.task_id, "t1")
        self.assertEqual(vm_state.dequeue().task.task_id, "t2")

    def test_start_updates_current_and_availability(self) -> None:
        vm_state = VirtualMachineExecutionState(vm=make_vm())
        vm_state.enqueue(task=make_task("t1"), duration=3.0, enqueued_at=0.0)

        running = vm_state.start_task(at_time=2.0)

        self.assertEqual(running.start_time, 2.0)
        self.assertEqual(running.finish_time, 5.0)
        self.assertEqual(vm_state.current_task.task.task_id, "t1")
        self.assertEqual(vm_state.availability_time, 5.0)
        self.assertEqual(vm_state.idle_intervals, [(0.0, 2.0)])

    def test_finish_updates_history_and_busy_interval(self) -> None:
        vm_state = VirtualMachineExecutionState(vm=make_vm())
        vm_state.enqueue(task=make_task("t1"), duration=3.0, enqueued_at=0.0)
        vm_state.start_task(at_time=0.0)

        record = vm_state.finish_task(at_time=3.0)

        self.assertEqual(record.task.task_id, "t1")
        self.assertEqual(vm_state.current_task, None)
        self.assertEqual(vm_state.busy_intervals, [(0.0, 3.0)])
        self.assertEqual(len(vm_state.task_history), 1)
        self.assertEqual(vm_state.availability_time, 3.0)


class TestVirtualMachineExecutionManagerIntegration(unittest.TestCase):
    def test_event_driven_enqueue_start_finish(self) -> None:
        sim = Simulation()
        manager = VirtualMachineExecutionManager.from_virtual_machines([make_vm("vm1", availability=0.0)])
        manager.register(sim)

        sim.schedule(
            Event(
                time=0.0,
                name="vm.enqueue",
                payload={"vm_id": "vm1", "task": make_task("t1"), "duration": 2.0},
            )
        )
        sim.schedule(
            Event(
                time=1.0,
                name="vm.enqueue",
                payload={"vm_id": "vm1", "task": make_task("t2"), "duration": 2.0},
            )
        )

        sim.run()
        vm_state = manager.get_vm_state("vm1")

        self.assertEqual([r.task.task_id for r in vm_state.task_history], ["t1", "t2"])
        self.assertEqual(vm_state.busy_intervals, [(0.0, 2.0), (2.0, 4.0)])
        self.assertEqual(vm_state.queue_length(), 0)
        self.assertEqual(vm_state.current_task, None)
        self.assertEqual(vm_state.availability_time, 4.0)

    def test_respects_initial_availability_time(self) -> None:
        sim = Simulation()
        manager = VirtualMachineExecutionManager.from_virtual_machines([make_vm("vm1", availability=3.0)])
        manager.register(sim)

        sim.schedule(
            Event(
                time=0.0,
                name="vm.enqueue",
                payload={"vm_id": "vm1", "task": make_task("t1"), "duration": 1.0},
            )
        )

        sim.run()
        vm_state = manager.get_vm_state("vm1")

        self.assertEqual(vm_state.task_history[0].start_time, 3.0)
        self.assertEqual(vm_state.task_history[0].finish_time, 4.0)


if __name__ == "__main__":
    unittest.main()

