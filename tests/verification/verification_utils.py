import math
from typing import Dict, List

from desim.framework.models import CloudConfiguration, Datacenter, PhysicalMachine, PowerProfile, ResourceCapacity, Task, VirtualMachine
from desim.framework.orchestrator import SimulationOrchestrator
from desim.algorithms.scheduling import Scheduler, SchedulingResult


TOLERANCE = 1e-6


class FixedScheduler(Scheduler):
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = dict(mapping)

    def schedule(self, tasks, virtual_machines):
        return SchedulingResult(task_to_vm=self.mapping)


def make_pm(machine_id: str = "pm1", base_power_watts: float = 100.0) -> PhysicalMachine:
    return PhysicalMachine(
        machine_id=machine_id,
        capacity=ResourceCapacity(cpu_mips=10000, memory_mb=65536, bandwidth_mbps=10000),
        base_power_watts=base_power_watts,
    )


def make_vm(
    vm_id: str,
    host_machine_id: str,
    cpu_mips: float = 1000.0,
    idle_watts: float = 20.0,
    max_watts: float = 60.0,
    availability_time: float = 0.0,
) -> VirtualMachine:
    return VirtualMachine(
        vm_id=vm_id,
        vm_type="small",
        host_machine_id=host_machine_id,
        capacity=ResourceCapacity(cpu_mips=cpu_mips, memory_mb=2048, bandwidth_mbps=100),
        power=PowerProfile(idle_watts=idle_watts, max_watts=max_watts),
        vcpu_count=2,
        availability_time=availability_time,
    )


def make_task(
    task_id: str,
    workload_mi: float,
    arrival_time: float,
    deadline: float,
    cpu_demand_mips: float,
) -> Task:
    return Task(
        task_id=task_id,
        workload_mi=workload_mi,
        arrival_time=arrival_time,
        deadline=deadline,
        cpu_demand_mips=cpu_demand_mips,
        memory_demand_mb=128,
        io_size_mb=10,
    )


def build_configuration(
    epoch_length: float,
    physical_machines: List[PhysicalMachine],
    virtual_machines: List[VirtualMachine],
    tasks: List[Task],
) -> Dict[str, object]:
    return {
        "epoch_length": epoch_length,
        "datacenter": {
            "datacenter_id": "dc1",
            "physical_machines": [
                {
                    "machine_id": pm.machine_id,
                    "cpu_mips": pm.capacity.cpu_mips,
                    "memory_mb": pm.capacity.memory_mb,
                    "bandwidth_mbps": pm.capacity.bandwidth_mbps,
                    "base_power_watts": pm.base_power_watts,
                }
                for pm in physical_machines
            ],
            "virtual_machines": [
                {
                    "vm_id": vm.vm_id,
                    "vm_type": vm.vm_type,
                    "host_machine_id": vm.host_machine_id,
                    "cpu_mips": vm.capacity.cpu_mips,
                    "memory_mb": vm.capacity.memory_mb,
                    "bandwidth_mbps": vm.capacity.bandwidth_mbps,
                    "idle_watts": vm.power.idle_watts,
                    "max_watts": vm.power.max_watts,
                    "vcpu_count": vm.vcpu_count,
                    "availability_time": vm.availability_time,
                }
                for vm in virtual_machines
            ],
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "workload_mi": task.workload_mi,
                "arrival_time": task.arrival_time,
                "deadline": task.deadline,
                "cpu_demand_mips": task.cpu_demand_mips,
                "memory_demand_mb": task.memory_demand_mb,
                "io_size_mb": task.io_size_mb,
            }
            for task in tasks
        ],
    }


def run_scenario(dataset: Dict[str, object], mapping: Dict[str, str]):
    orchestrator = SimulationOrchestrator(scheduler=FixedScheduler(mapping))
    state = orchestrator.run(dataset)
    return state.get("metrics")


def linear_dynamic_energy(idle_watts: float, max_watts: float, utilization: float, duration: float) -> float:
    alpha = max_watts - idle_watts
    beta = 0.0
    return (alpha * utilization + beta * utilization * utilization) * duration


def exponential_sla_penalty(completion_time: float, deadline: float, lambda_: float = 1.0, theta: float = 1.0, eta_max: float = 2.0) -> float:
    violation = max(0.0, completion_time - deadline)
    if violation == 0.0:
        return 0.0
    eta = violation / deadline
    if eta <= eta_max:
        return lambda_ * (math.exp(theta * eta) - 1.0)
    kappa = lambda_ * theta * math.exp(theta * eta_max)
    return lambda_ * (math.exp(theta * eta_max) - 1.0) + kappa * (eta - eta_max)


def jain_index(values: List[float]) -> float:
    total = sum(values)
    squares = sum(v * v for v in values)
    if total == 0.0 and squares == 0.0:
        return 1.0
    return (total * total) / (len(values) * squares)

