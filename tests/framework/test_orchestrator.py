import json
from pathlib import Path
import tempfile
import unittest

from desim.framework.orchestrator import OrchestratorRuntime, SimulationOrchestrator
from desim.algorithms.scheduling import Scheduler, SchedulingResult
from desim.framework.simulation import Simulation


class FixedScheduler(Scheduler):
    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def schedule(self, tasks, virtual_machines):
        return SchedulingResult(task_to_vm=self.mapping)


def dataset_fixture() -> dict:
    return {
        "epoch_length": 20,
        "datacenter": {
            "datacenter_id": "dc1",
            "physical_machines": [
                {
                    "machine_id": "pm1",
                    "cpu_mips": 4000,
                    "memory_mb": 16384,
                    "bandwidth_mbps": 1000,
                    "base_power_watts": 120,
                }
            ],
            "virtual_machines": [
                {
                    "vm_id": "vm1",
                    "vm_type": "small",
                    "host_machine_id": "pm1",
                    "cpu_mips": 1000,
                    "memory_mb": 2048,
                    "bandwidth_mbps": 100,
                    "idle_watts": 20,
                    "max_watts": 60,
                    "vcpu_count": 2,
                    "availability_time": 0,
                },
                {
                    "vm_id": "vm2",
                    "vm_type": "small",
                    "host_machine_id": "pm1",
                    "cpu_mips": 1000,
                    "memory_mb": 2048,
                    "bandwidth_mbps": 100,
                    "idle_watts": 20,
                    "max_watts": 60,
                    "vcpu_count": 2,
                    "availability_time": 0,
                },
            ],
        },
        "tasks": [
            {
                "task_id": "t1",
                "workload_mi": 1000,
                "arrival_time": 0,
                "deadline": 10,
                "cpu_demand_mips": 100,
                "memory_demand_mb": 128,
                "io_size_mb": 10,
            },
            {
                "task_id": "t2",
                "workload_mi": 2000,
                "arrival_time": 1,
                "deadline": 10,
                "cpu_demand_mips": 100,
                "memory_demand_mb": 128,
                "io_size_mb": 10,
            },
        ],
    }


class TestSimulationOrchestrator(unittest.TestCase):
    def test_load_dataset_from_dict(self) -> None:
        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler({}))
        data = dataset_fixture()
        self.assertEqual(orchestrator.load_dataset(data)["epoch_length"], 20)

    def test_load_dataset_from_json_file(self) -> None:
        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler({}))
        data = dataset_fixture()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f)

            loaded = orchestrator.load_dataset(path)
            self.assertEqual(loaded["datacenter"]["datacenter_id"], "dc1")

    def test_create_cloud_configuration(self) -> None:
        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler({}))
        cfg = orchestrator.create_cloud_configuration(dataset_fixture())
        self.assertEqual(cfg.datacenter.datacenter_id, "dc1")
        self.assertEqual(len(cfg.datacenter.physical_machines), 1)
        self.assertEqual(len(cfg.datacenter.virtual_machines), 2)
        self.assertEqual(len(cfg.tasks), 2)

    def test_insert_events_rejects_unknown_vm(self) -> None:
        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler({}))
        cfg = orchestrator.create_cloud_configuration(dataset_fixture())
        sim = Simulation()

        with self.assertRaises(ValueError):
            orchestrator.insert_events(
                simulation=sim,
                configuration=cfg,
                assignment=SchedulingResult({"t1": "vm-x"}),
            )

    def test_insert_events_rejects_memory_infeasible_assignment(self) -> None:
        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler({}))
        data = dataset_fixture()
        data["tasks"][0]["memory_demand_mb"] = 4096
        cfg = orchestrator.create_cloud_configuration(data)
        sim = Simulation()

        with self.assertRaises(ValueError):
            orchestrator.insert_events(
                simulation=sim,
                configuration=cfg,
                assignment=SchedulingResult({"t1": "vm1"}),
            )

    def test_insert_events_allows_sequential_tasks_when_each_fits_vm_memory(self) -> None:
        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler({}))
        data = dataset_fixture()
        data["tasks"][0]["memory_demand_mb"] = 1800
        data["tasks"][1]["memory_demand_mb"] = 1800
        data["datacenter"]["virtual_machines"][0]["memory_mb"] = 2048

        cfg = orchestrator.create_cloud_configuration(data)
        sim = Simulation()

        orchestrator.insert_events(
            simulation=sim,
            configuration=cfg,
            assignment=SchedulingResult({"t1": "vm1", "t2": "vm1"}),
        )

        self.assertEqual(len(sim.event_queue), 2)

    def test_run_executes_and_returns_state(self) -> None:
        mapping = {"t1": "vm1", "t2": "vm2"}
        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler(mapping))

        state = orchestrator.run(dataset_fixture())

        self.assertEqual(state.get("assignment").task_to_vm, mapping)
        runtime = state.get("orchestrator_runtime")
        self.assertIsInstance(runtime, OrchestratorRuntime)

        vm_exec = runtime.vm_execution
        vm1_history = vm_exec.get_vm_state("vm1").task_history
        vm2_history = vm_exec.get_vm_state("vm2").task_history

        self.assertEqual([r.task.task_id for r in vm1_history], ["t1"])
        self.assertEqual([r.task.task_id for r in vm2_history], ["t2"])

        utilization = runtime.utilization
        trace_vm1 = utilization.snapshot().traces["vm1"]
        self.assertTrue(len(trace_vm1.intervals) > 0)

    def test_run_stops_at_epoch_limit(self) -> None:
        data = dataset_fixture()
        data["epoch_length"] = 0.5
        mapping = {"t1": "vm1", "t2": "vm1"}
        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler(mapping))

        state = orchestrator.run(data)
        vm_exec = state.get("vm_execution")
        history = vm_exec.get_vm_state("vm1").task_history

        # t1 duration is 1.0s, so at epoch 0.5 it should not have finished yet.
        self.assertEqual(len(history), 0)


if __name__ == "__main__":
    unittest.main()

