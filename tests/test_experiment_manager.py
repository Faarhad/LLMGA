from pathlib import Path
import tempfile
import unittest

from desim.experiment_manager import ExperimentManager, ExperimentRunner
from desim.metrics_collector import MetricsSnapshot
from desim.random_benchmark import RandomScheduleBenchmarkRunner


class BenchmarkRunnerAdapter(ExperimentRunner):
    def __init__(self, dataset_source: dict):
        self.dataset_source = dataset_source
        self.runner = RandomScheduleBenchmarkRunner(random_seed=5, uncertainty_repeats=3, uncertainty_noise_std=0.01)

    def run(self, seed: int) -> MetricsSnapshot:
        # Use the seed to create repeatable benchmark runs while keeping the manager interface generic.
        local_runner = RandomScheduleBenchmarkRunner(random_seed=seed, uncertainty_repeats=3, uncertainty_noise_std=0.01)
        stats = local_runner.run(self.dataset_source, k=1)
        return stats.runs[0].metrics


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
                }
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
            }
        ],
    }


class TestExperimentManager(unittest.TestCase):
    def test_run_multiple_experiments_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(
                runner=BenchmarkRunnerAdapter(dataset_fixture()),
                output_dir=tmpdir,
                base_seed=11,
            )

            runs = manager.run(run_count=3)
            summary = manager.summarize()
            csv_path = manager.export_csv()
            json_runs_path = manager.save_runs_json()
            json_summary_path = manager.save_summary_json()
            plots = manager.export_plots()

            self.assertEqual(len(runs), 3)
            self.assertEqual(summary.run_count, 3)
            self.assertEqual(len(summary.seeds), 3)
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_runs_path.exists())
            self.assertTrue(json_summary_path.exists())
            self.assertEqual(len(plots), 5)
            for plot in plots:
                self.assertTrue(Path(plot.path).exists())

    def test_run_count_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(runner=BenchmarkRunnerAdapter(dataset_fixture()), output_dir=tmpdir)
            with self.assertRaises(ValueError):
                manager.run(run_count=0)

    def test_summarize_without_runs_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(runner=BenchmarkRunnerAdapter(dataset_fixture()), output_dir=tmpdir)
            with self.assertRaises(ValueError):
                manager.summarize()


if __name__ == "__main__":
    unittest.main()
