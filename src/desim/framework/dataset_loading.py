import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


class DatasetValidationError(ValueError):
    pass


@dataclass
class DatasetValidator:
    """Validates raw dataset structure before simulation parsing."""

    def validate(self, dataset: Dict[str, Any]) -> None:
        self._require_keys(dataset, ["epoch_length", "datacenter", "tasks"], "root")

        datacenter = dataset["datacenter"]
        if not isinstance(datacenter, dict):
            raise DatasetValidationError("datacenter must be an object")

        self._require_keys(
            datacenter,
            ["datacenter_id", "physical_machines", "virtual_machines"],
            "datacenter",
        )

        self._validate_number(dataset["epoch_length"], "epoch_length")

        if not isinstance(dataset["tasks"], list):
            raise DatasetValidationError("tasks must be a list")
        if not isinstance(datacenter["physical_machines"], list):
            raise DatasetValidationError("datacenter.physical_machines must be a list")
        if not isinstance(datacenter["virtual_machines"], list):
            raise DatasetValidationError("datacenter.virtual_machines must be a list")

        self._validate_pm_rows(datacenter["physical_machines"])
        self._validate_vm_rows(datacenter["virtual_machines"])
        self._validate_task_rows(dataset["tasks"])

        self._validate_id_uniqueness(datacenter["physical_machines"], "machine_id", "physical machines")
        self._validate_id_uniqueness(datacenter["virtual_machines"], "vm_id", "virtual machines")
        self._validate_id_uniqueness(dataset["tasks"], "task_id", "tasks")

        pm_ids = {row["machine_id"] for row in datacenter["physical_machines"]}
        for vm in datacenter["virtual_machines"]:
            if vm["host_machine_id"] not in pm_ids:
                raise DatasetValidationError(
                    f"vm host_machine_id '{vm['host_machine_id']}' does not reference an existing machine_id"
                )

    def _validate_pm_rows(self, rows: List[Dict[str, Any]]) -> None:
        required = ["machine_id", "cpu_mips", "memory_mb", "bandwidth_mbps", "base_power_watts"]
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise DatasetValidationError(f"physical_machines[{i}] must be an object")
            self._require_keys(row, required, f"physical_machines[{i}]")
            self._validate_number(row["cpu_mips"], f"physical_machines[{i}].cpu_mips")
            self._validate_number(row["memory_mb"], f"physical_machines[{i}].memory_mb")
            self._validate_number(row["bandwidth_mbps"], f"physical_machines[{i}].bandwidth_mbps")
            self._validate_number(row["base_power_watts"], f"physical_machines[{i}].base_power_watts")

    def _validate_vm_rows(self, rows: List[Dict[str, Any]]) -> None:
        required = [
            "vm_id",
            "vm_type",
            "host_machine_id",
            "cpu_mips",
            "memory_mb",
            "bandwidth_mbps",
            "idle_watts",
            "max_watts",
            "vcpu_count",
        ]
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise DatasetValidationError(f"virtual_machines[{i}] must be an object")
            self._require_keys(row, required, f"virtual_machines[{i}]")
            self._validate_number(row["cpu_mips"], f"virtual_machines[{i}].cpu_mips")
            self._validate_number(row["memory_mb"], f"virtual_machines[{i}].memory_mb")
            self._validate_number(row["bandwidth_mbps"], f"virtual_machines[{i}].bandwidth_mbps")
            self._validate_number(row["idle_watts"], f"virtual_machines[{i}].idle_watts")
            self._validate_number(row["max_watts"], f"virtual_machines[{i}].max_watts")
            self._validate_number(row["vcpu_count"], f"virtual_machines[{i}].vcpu_count")
            if "availability_time" in row:
                self._validate_number(row["availability_time"], f"virtual_machines[{i}].availability_time")

    def _validate_task_rows(self, rows: List[Dict[str, Any]]) -> None:
        required = [
            "task_id",
            "workload_mi",
            "arrival_time",
            "deadline",
            "cpu_demand_mips",
            "memory_demand_mb",
            "io_size_mb",
        ]
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise DatasetValidationError(f"tasks[{i}] must be an object")
            self._require_keys(row, required, f"tasks[{i}]")
            self._validate_number(row["workload_mi"], f"tasks[{i}].workload_mi")
            self._validate_number(row["arrival_time"], f"tasks[{i}].arrival_time")
            self._validate_number(row["deadline"], f"tasks[{i}].deadline")
            self._validate_number(row["cpu_demand_mips"], f"tasks[{i}].cpu_demand_mips")
            self._validate_number(row["memory_demand_mb"], f"tasks[{i}].memory_demand_mb")
            self._validate_number(row["io_size_mb"], f"tasks[{i}].io_size_mb")

    @staticmethod
    def _validate_id_uniqueness(rows: Iterable[Dict[str, Any]], key: str, label: str) -> None:
        values = [str(r[key]) for r in rows]
        if len(set(values)) != len(values):
            raise DatasetValidationError(f"duplicate {key} found in {label}")

    @staticmethod
    def _require_keys(obj: Dict[str, Any], required: List[str], ctx: str) -> None:
        missing = [k for k in required if k not in obj]
        if missing:
            raise DatasetValidationError(f"missing keys in {ctx}: {', '.join(missing)}")

    @staticmethod
    def _validate_number(value: Any, field_name: str) -> None:
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(f"{field_name} must be numeric") from exc


@dataclass
class DatasetParser:
    """Parses JSON and CSV datasets into a common dictionary structure."""

    def parse_json(self, file_path: Path) -> Dict[str, Any]:
        # Accept JSON files saved with or without UTF-8 BOM.
        with file_path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    def parse_csv_bundle(self, folder: Path) -> Dict[str, Any]:
        datacenter_rows = self._read_csv_rows(folder / "datacenter.csv")
        pm_rows = self._read_csv_rows(folder / "physical_machines.csv")
        vm_rows = self._read_csv_rows(folder / "virtual_machines.csv")
        task_rows = self._read_csv_rows(folder / "tasks.csv")

        if len(datacenter_rows) != 1:
            raise DatasetValidationError("datacenter.csv must contain exactly one row")

        dc = datacenter_rows[0]
        return {
            "epoch_length": dc["epoch_length"],
            "datacenter": {
                "datacenter_id": dc["datacenter_id"],
                "physical_machines": pm_rows,
                "virtual_machines": vm_rows,
            },
            "tasks": task_rows,
        }

    @staticmethod
    def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise DatasetValidationError(f"required CSV file not found: {path.name}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise DatasetValidationError(f"CSV file has no header: {path.name}")
            return [dict(row) for row in reader]


@dataclass
class DatasetLoader:
    """Loads and validates datasets from dict, JSON, or CSV bundle."""

    parser: DatasetParser = field(default_factory=DatasetParser)
    validator: DatasetValidator = field(default_factory=DatasetValidator)

    def load(self, source: Dict[str, Any] | str | Path) -> Dict[str, Any]:
        if isinstance(source, dict):
            dataset = source
        else:
            path = Path(source)
            if path.is_dir():
                dataset = self.parser.parse_csv_bundle(path)
            elif path.suffix.lower() == ".json":
                dataset = self.parser.parse_json(path)
            elif path.suffix.lower() == ".csv":
                dataset = self.parser.parse_csv_bundle(path.parent)
            else:
                raise DatasetValidationError("unsupported dataset source; use dict, .json, or CSV bundle folder")

        self.validator.validate(dataset)
        return dataset
