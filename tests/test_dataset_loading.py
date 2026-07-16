import json
from pathlib import Path
import tempfile
import unittest

from desim.dataset_loading import DatasetLoader, DatasetValidationError


def valid_dataset_dict() -> dict:
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
                }
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
            }
        ],
    }


class TestDatasetLoading(unittest.TestCase):
    def test_load_from_dict(self) -> None:
        loader = DatasetLoader()
        data = loader.load(valid_dataset_dict())
        self.assertEqual(data["datacenter"]["datacenter_id"], "dc1")

    def test_load_from_json(self) -> None:
        loader = DatasetLoader()
        data = valid_dataset_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f)

            loaded = loader.load(path)
            self.assertEqual(loaded["epoch_length"], 20)

    def test_load_from_csv_bundle_folder(self) -> None:
        loader = DatasetLoader()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "datacenter.csv").write_text(
                "datacenter_id,epoch_length\n"
                "dc1,20\n",
                encoding="utf-8",
            )
            (root / "physical_machines.csv").write_text(
                "machine_id,cpu_mips,memory_mb,bandwidth_mbps,base_power_watts\n"
                "pm1,4000,16384,1000,120\n",
                encoding="utf-8",
            )
            (root / "virtual_machines.csv").write_text(
                "vm_id,vm_type,host_machine_id,cpu_mips,memory_mb,bandwidth_mbps,idle_watts,max_watts,vcpu_count,availability_time\n"
                "vm1,small,pm1,1000,2048,100,20,60,2,0\n",
                encoding="utf-8",
            )
            (root / "tasks.csv").write_text(
                "task_id,workload_mi,arrival_time,deadline,cpu_demand_mips,memory_demand_mb,io_size_mb\n"
                "t1,1000,0,10,100,128,10\n",
                encoding="utf-8",
            )

            loaded = loader.load(root)
            self.assertEqual(loaded["datacenter"]["datacenter_id"], "dc1")
            self.assertEqual(len(loaded["tasks"]), 1)

    def test_missing_required_field_fails_validation(self) -> None:
        loader = DatasetLoader()
        broken = valid_dataset_dict()
        del broken["tasks"][0]["deadline"]

        with self.assertRaises(DatasetValidationError):
            loader.load(broken)

    def test_unknown_vm_host_fails_validation(self) -> None:
        loader = DatasetLoader()
        broken = valid_dataset_dict()
        broken["datacenter"]["virtual_machines"][0]["host_machine_id"] = "pm-x"

        with self.assertRaises(DatasetValidationError):
            loader.load(broken)


if __name__ == "__main__":
    unittest.main()
