import unittest

from desim.research.metrics_collector import MetricsCollector
from desim.framework.models import (
    CloudConfiguration,
    Datacenter,
    PhysicalMachine,
    PowerProfile,
    ResourceCapacity,
    Task,
    VirtualMachine,
)
from desim.framework.orchestrator import SimulationOrchestrator
from desim.algorithms.scheduling import Scheduler, SchedulingResult
from desim.framework.utilization import UtilizationTracker


class FixedScheduler(Scheduler):
    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def schedule(self, waiting_tasks, vm_states):
        waiting_ids = {task.task_id for task in waiting_tasks}
        filtered = {task_id: vm_id for task_id, vm_id in self.mapping.items() if task_id in waiting_ids}
        return SchedulingResult(task_to_vm=filtered)


def fixture_configuration() -> CloudConfiguration:
    pm = PhysicalMachine(
        machine_id="pm1",
        capacity=ResourceCapacity(cpu_mips=10000, memory_mb=65536, bandwidth_mbps=10000),
        base_power_watts=100.0,
    )
    vm = VirtualMachine(
        vm_id="vm1",
        vm_type="small",
        host_machine_id="pm1",
        capacity=ResourceCapacity(cpu_mips=1000, memory_mb=2048, bandwidth_mbps=100),
        power=PowerProfile(idle_watts=20.0, max_watts=60.0),
        vcpu_count=2,
        availability_time=1.0,
    )
    t1 = Task("t1", workload_mi=1000, arrival_time=0.0, deadline=1.5, cpu_demand_mips=500, memory_demand_mb=128, io_size_mb=10)
    t2 = Task("t2", workload_mi=1000, arrival_time=0.0, deadline=2.5, cpu_demand_mips=500, memory_demand_mb=128, io_size_mb=10)
    dc = Datacenter(datacenter_id="dc1", physical_machines=[pm], virtual_machines=[vm])
    return CloudConfiguration(datacenter=dc, tasks=[t1, t2], epoch_length=5.0)


class TestMetricsCollector(unittest.TestCase):
    def test_collector_computes_requested_metrics(self) -> None:
        cfg = fixture_configuration()

        dataset = {
            "epoch_length": cfg.epoch_length,
            "datacenter": {
                "datacenter_id": "dc1",
                "physical_machines": [
                    {
                        "machine_id": "pm1",
                        "cpu_mips": 10000,
                        "memory_mb": 65536,
                        "bandwidth_mbps": 10000,
                        "base_power_watts": 100.0,
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
                        "idle_watts": 20.0,
                        "max_watts": 60.0,
                        "vcpu_count": 2,
                        "availability_time": 1.0,
                    }
                ],
            },
            "tasks": [
                {
                    "task_id": "t1",
                    "workload_mi": 1000,
                    "arrival_time": 0.0,
                    "deadline": 1.5,
                    "cpu_demand_mips": 500,
                    "memory_demand_mb": 128,
                    "io_size_mb": 10,
                },
                {
                    "task_id": "t2",
                    "workload_mi": 1000,
                    "arrival_time": 0.0,
                    "deadline": 2.5,
                    "cpu_demand_mips": 500,
                    "memory_demand_mb": 128,
                    "io_size_mb": 10,
                },
            ],
        }

        orchestrator = SimulationOrchestrator(scheduler=FixedScheduler({"t1": "vm1", "t2": "vm1"}))
        state = orchestrator.run(dataset)

        snapshot = state.get("metrics")

        self.assertAlmostEqual(snapshot.makespan, 3.0)
        self.assertAlmostEqual(snapshot.throughput, 2.0 / 3.0)
        self.assertAlmostEqual(snapshot.average_waiting_time, 1.5)
        self.assertAlmostEqual(snapshot.average_response_time, 2.5)

        # Energy from interval integration + PM base over epoch.
        self.assertAlmostEqual(snapshot.energy.total_vm_energy, 140.0)
        self.assertAlmostEqual(snapshot.energy.total_pm_base_energy, 500.0)
        self.assertAlmostEqual(snapshot.energy.total_energy, 640.0)

        self.assertTrue(snapshot.sla.aggregate_penalty >= 0.0)
        self.assertTrue(snapshot.fairness.combined_fairness >= 0.0)

        self.assertAlmostEqual(snapshot.average_utilization_per_vm["vm1"], 0.2)
        self.assertAlmostEqual(snapshot.cpu_occupancy, 0.2)

        # Fitness is computed and scalar.
        self.assertTrue(snapshot.fitness > 0.0)

    def test_collector_observes_events_directly(self) -> None:
        cfg = fixture_configuration()
        utilization = UtilizationTracker.from_virtual_machines(cfg.datacenter.virtual_machines)
        collector = MetricsCollector(configuration=cfg, utilization_tracker=utilization)

        self.assertEqual(collector._finished_tasks, 0)


if __name__ == "__main__":
    unittest.main()

