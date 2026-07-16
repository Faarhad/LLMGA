from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ResourceCapacity:
    """Generic resource capacity container."""

    cpu_mips: float
    memory_mb: float
    bandwidth_mbps: float

    def __post_init__(self) -> None:
        if self.cpu_mips < 0 or self.memory_mb < 0 or self.bandwidth_mbps < 0:
            raise ValueError("resource capacities must be >= 0")


@dataclass(frozen=True)
class PowerProfile:
    """Power profile for machines and virtual machines."""

    idle_watts: float
    max_watts: float

    def __post_init__(self) -> None:
        if self.idle_watts < 0 or self.max_watts < 0:
            raise ValueError("power values must be >= 0")
        if self.max_watts < self.idle_watts:
            raise ValueError("max_watts must be >= idle_watts")


@dataclass(frozen=True)
class PhysicalMachine:
    """Physical machine state only."""

    machine_id: str
    capacity: ResourceCapacity
    base_power_watts: float
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.machine_id:
            raise ValueError("machine_id must be non-empty")
        if self.base_power_watts < 0:
            raise ValueError("base_power_watts must be >= 0")


@dataclass(frozen=True)
class VirtualMachine:
    """Virtual machine state only."""

    vm_id: str
    vm_type: str
    host_machine_id: str
    capacity: ResourceCapacity
    power: PowerProfile
    vcpu_count: int
    availability_time: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vm_id:
            raise ValueError("vm_id must be non-empty")
        if not self.vm_type:
            raise ValueError("vm_type must be non-empty")
        if not self.host_machine_id:
            raise ValueError("host_machine_id must be non-empty")
        if self.vcpu_count <= 0:
            raise ValueError("vcpu_count must be > 0")
        if self.availability_time < 0:
            raise ValueError("availability_time must be >= 0")


@dataclass(frozen=True)
class Task:
    """Task state only."""

    task_id: str
    workload_mi: float
    arrival_time: float
    deadline: float
    cpu_demand_mips: float
    memory_demand_mb: float
    io_size_mb: float
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.workload_mi < 0:
            raise ValueError("workload_mi must be >= 0")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be >= 0")
        if self.deadline < 0:
            raise ValueError("deadline must be >= 0")
        if self.cpu_demand_mips < 0 or self.memory_demand_mb < 0 or self.io_size_mb < 0:
            raise ValueError("task demands must be >= 0")


@dataclass(frozen=True)
class Datacenter:
    """Datacenter state, composed of machines and VMs."""

    datacenter_id: str
    physical_machines: List[PhysicalMachine]
    virtual_machines: List[VirtualMachine]
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.datacenter_id:
            raise ValueError("datacenter_id must be non-empty")

        pm_ids = {pm.machine_id for pm in self.physical_machines}
        if len(pm_ids) != len(self.physical_machines):
            raise ValueError("physical machine ids must be unique")

        vm_ids = {vm.vm_id for vm in self.virtual_machines}
        if len(vm_ids) != len(self.virtual_machines):
            raise ValueError("virtual machine ids must be unique")

        for vm in self.virtual_machines:
            if vm.host_machine_id not in pm_ids:
                raise ValueError("virtual machine host_machine_id must reference an existing physical machine")


@dataclass(frozen=True)
class CloudConfiguration:
    """Top-level cloud state for simulation input."""

    datacenter: Datacenter
    tasks: List[Task]
    epoch_length: float
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epoch_length <= 0:
            raise ValueError("epoch_length must be > 0")

        task_ids = {task.task_id for task in self.tasks}
        if len(task_ids) != len(self.tasks):
            raise ValueError("task ids must be unique")
