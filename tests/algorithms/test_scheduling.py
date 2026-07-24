import unittest

from desim.framework.models import PowerProfile, ResourceCapacity, Task, VirtualMachine
from desim.algorithms.scheduling import (
    GeneticAlgorithmScheduler,
    RandomScheduler,
    Scheduler,
    SchedulingResult,
    VmStateView,
)


class _DummyScheduler(Scheduler):
    def schedule(self, waiting_tasks, vm_states):
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


def make_vm_states(vms: list[VirtualMachine]) -> list[VmStateView]:
    return [VmStateView(vm=vm, availability_time=vm.availability_time) for vm in vms]


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
        vm_states = make_vm_states([make_vm("vm1"), make_vm("vm2")])

        result = scheduler.schedule(tasks, vm_states)

        self.assertEqual(set(result.task_to_vm.keys()), {"t1", "t2", "t3"})
        for vm_id in result.task_to_vm.values():
            self.assertIn(vm_id, {"vm1", "vm2"})

    def test_seed_makes_result_repeatable(self) -> None:
        tasks = [make_task("t1"), make_task("t2"), make_task("t3")]
        vm_states = make_vm_states([make_vm("vm1"), make_vm("vm2")])

        s1 = RandomScheduler(seed=42)
        s2 = RandomScheduler(seed=42)

        r1 = s1.schedule(tasks, vm_states)
        r2 = s2.schedule(tasks, vm_states)

        self.assertEqual(r1.task_to_vm, r2.task_to_vm)

    def test_no_vm_with_tasks_rejected(self) -> None:
        scheduler = RandomScheduler()
        with self.assertRaises(ValueError):
            scheduler.schedule([make_task("t1")], [])

    def test_no_tasks_returns_empty_mapping(self) -> None:
        scheduler = RandomScheduler()
        result = scheduler.schedule([], make_vm_states([make_vm("vm1")]))
        self.assertEqual(result.task_to_vm, {})

    def test_rejects_when_task_has_no_feasible_vm(self) -> None:
        scheduler = RandomScheduler(seed=1)
        infeasible_task = Task(
            task_id="t_bad",
            workload_mi=1000,
            arrival_time=0,
            deadline=10,
            cpu_demand_mips=1200,
            memory_demand_mb=3000,
            io_size_mb=10,
        )
        vm_states = make_vm_states([make_vm("vm1"), make_vm("vm2")])

        with self.assertRaises(ValueError):
            scheduler.schedule([infeasible_task], vm_states)


class TestGeneticAlgorithmScheduler(unittest.TestCase):
    def test_returns_complete_feasible_mapping(self) -> None:
        scheduler = GeneticAlgorithmScheduler(
            population_size=20,
            generations=25,
            seed=13,
        )
        tasks = [
            make_task("t1"),
            make_task("t2"),
            make_task("t3"),
        ]
        vm_states = make_vm_states([
            make_vm("vm1"),
            make_vm("vm2"),
        ])

        result = scheduler.schedule(tasks, vm_states)

        self.assertEqual(set(result.task_to_vm.keys()), {"t1", "t2", "t3"})
        self.assertTrue(all(vm in {"vm1", "vm2"} for vm in result.task_to_vm.values()))

    def test_seed_makes_result_repeatable(self) -> None:
        tasks = [make_task("t1"), make_task("t2"), make_task("t3"), make_task("t4")]
        vm_states = make_vm_states([make_vm("vm1"), make_vm("vm2")])

        s1 = GeneticAlgorithmScheduler(population_size=24, generations=30, seed=99)
        s2 = GeneticAlgorithmScheduler(population_size=24, generations=30, seed=99)

        r1 = s1.schedule(tasks, vm_states)
        r2 = s2.schedule(tasks, vm_states)

        self.assertEqual(r1.task_to_vm, r2.task_to_vm)

    def test_rejects_when_task_has_no_feasible_vm(self) -> None:
        scheduler = GeneticAlgorithmScheduler(seed=5)
        infeasible_task = Task(
            task_id="t_bad",
            workload_mi=1000,
            arrival_time=0,
            deadline=10,
            cpu_demand_mips=100,
            memory_demand_mb=5000,
            io_size_mb=10,
        )
        vm_states = make_vm_states([make_vm("vm1"), make_vm("vm2")])

        with self.assertRaises(ValueError):
            scheduler.schedule([infeasible_task], vm_states)


if __name__ == "__main__":
    unittest.main()

