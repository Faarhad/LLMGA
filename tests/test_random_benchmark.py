import unittest

from desim.random_benchmark import RandomScheduleBenchmarkRunner


def dataset_fixture() -> dict:
    return {
        "epoch_length": 10,
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
                "deadline": 5,
                "cpu_demand_mips": 100,
                "memory_demand_mb": 128,
                "io_size_mb": 10,
            },
            {
                "task_id": "t2",
                "workload_mi": 1000,
                "arrival_time": 0,
                "deadline": 5,
                "cpu_demand_mips": 100,
                "memory_demand_mb": 128,
                "io_size_mb": 10,
            },
        ],
    }


class TestRandomScheduleBenchmarkRunner(unittest.TestCase):
    def test_runs_k_random_schedules_and_returns_maxima(self) -> None:
        runner = RandomScheduleBenchmarkRunner(random_seed=7, uncertainty_repeats=3, uncertainty_noise_std=0.01)
        stats = runner.run(dataset_fixture(), k=10)

        self.assertEqual(stats.sample_count, 10)
        self.assertEqual(len(stats.runs), 10)

        self.assertAlmostEqual(stats.energy_max_rand, max(r.energy_total for r in stats.runs))
        self.assertAlmostEqual(stats.duration_max_rand, max(r.duration for r in stats.runs))
        self.assertAlmostEqual(stats.uncertainty_max_rand, max(r.uncertainty_spread for r in stats.runs))
        self.assertAlmostEqual(stats.sla_objective_max_rand, max(r.sla_objective for r in stats.runs))

    def test_invalid_k_rejected(self) -> None:
        runner = RandomScheduleBenchmarkRunner()
        with self.assertRaises(ValueError):
            runner.run(dataset_fixture(), k=0)


if __name__ == "__main__":
    unittest.main()
