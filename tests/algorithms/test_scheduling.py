import unittest

from desim.framework.models import PowerProfile, ResourceCapacity, Task, VirtualMachine
from desim.algorithms.scheduling import RandomScheduler, Scheduler, SchedulingResult


class _DummyScheduler(Scheduler):
    def schedule(self, tasks, virtual_machines):
        return SchedulingResult({})


def make_vm(vm_id: str) -> VirtualMachine:
    return VirtualMachine(
        vm_id=vm_id,
        vm_type="small",
        host_machine_id="pm1",
        capacity=ResourceCapacity(1000, 2048, 100),
        power=PowerProfile(20, 60),
        vcpu_count=2,
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


class TestSchedulingResult(unittest.TestCase):
    def test_valid_mapping(self) -> None:
        result = SchedulingResult(task_to_vm={"t1": "vm1"})
        self.assertEqual(result.task_to_vm["t1"], "vm1")

    def test_empty_task_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SchedulingResult(task_to_vm={"": "vm1"})


class TestSchedulerABC(unittest.TestCase):
    def test_abstract_cannot_instantiate(self) -> None:
        with self.assertRaises(TypeError):
            Scheduler()

    def test_dummy_scheduler_works(self) -> None:
        s = _DummyScheduler()
        self.assertIsInstance(s.schedule([], []).task_to_vm, dict)


class TestRandomScheduler(unittest.TestCase):
    def test_returns_complete_task_mapping(self) -> None:
        scheduler = RandomScheduler(seed=7)
        tasks = [make_task("t1"), make_task("t2"), make_task("t3")]
        vms = [make_vm("vm1"), make_vm("vm2")]

        result = scheduler.schedule(tasks, vms)

        self.assertEqual(set(result.task_to_vm.keys()), {"t1", "t2", "t3"})
        for vm_id in result.task_to_vm.values():
            self.assertIn(vm_id, {"vm1", "vm2"})

    def test_seed_makes_result_repeatable(self) -> None:
        tasks = [make_task("t1"), make_task("t2"), make_task("t3")]
        vms = [make_vm("vm1"), make_vm("vm2")]

        s1 = RandomScheduler(seed=42)
        s2 = RandomScheduler(seed=42)

        r1 = s1.schedule(tasks, vms)
        r2 = s2.schedule(tasks, vms)

        self.assertEqual(r1.task_to_vm, r2.task_to_vm)

    def test_no_vm_with_tasks_rejected(self) -> None:
        scheduler = RandomScheduler()
        with self.assertRaises(ValueError):
            scheduler.schedule([make_task("t1")], [])

    def test_no_tasks_returns_empty_mapping(self) -> None:
        scheduler = RandomScheduler()
        result = scheduler.schedule([], [make_vm("vm1")])
        self.assertEqual(result.task_to_vm, {})


if __name__ == "__main__":
    unittest.main()

