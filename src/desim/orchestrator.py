from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .dataset_loading import DatasetLoader
from .event import Event
from .models import (
    CloudConfiguration,
    Datacenter,
    PhysicalMachine,
    PowerProfile,
    ResourceCapacity,
    Task,
    VirtualMachine,
)
from .metrics_collector import MetricsCollector, MetricsSnapshot
from .scheduling import Scheduler, SchedulingResult
from .simulation import Simulation
from .state import SimulationState
from .timing_metrics import TimingMetricsCollector
from .utilization import UtilizationTracker
from .vm_execution import VirtualMachineExecutionManager


@dataclass
class OrchestratorRuntime:
    """Runtime objects produced by a simulation orchestration run."""

    configuration: CloudConfiguration
    simulation: Simulation
    vm_execution: VirtualMachineExecutionManager
    timing_metrics: TimingMetricsCollector
    utilization: UtilizationTracker
    metrics: MetricsSnapshot
    assignment: SchedulingResult


class SimulationOrchestrator:
    """Coordinates load-build-schedule-run workflow for the simulator."""

    def __init__(self, scheduler: Scheduler, dataset_loader: DatasetLoader | None = None) -> None:
        self.scheduler = scheduler
        self.dataset_loader = dataset_loader or DatasetLoader()

    def load_dataset(self, dataset_source: Dict[str, Any] | str | Path) -> Dict[str, Any]:
        return self.dataset_loader.load(dataset_source)

    def create_physical_machines(self, dataset: Dict[str, Any]) -> List[PhysicalMachine]:
        raw_items = dataset["datacenter"]["physical_machines"]
        pms: List[PhysicalMachine] = []
        for item in raw_items:
            pms.append(
                PhysicalMachine(
                    machine_id=item["machine_id"],
                    capacity=ResourceCapacity(
                        cpu_mips=float(item["cpu_mips"]),
                        memory_mb=float(item["memory_mb"]),
                        bandwidth_mbps=float(item["bandwidth_mbps"]),
                    ),
                    base_power_watts=float(item["base_power_watts"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return pms

    def create_virtual_machines(self, dataset: Dict[str, Any]) -> List[VirtualMachine]:
        raw_items = dataset["datacenter"]["virtual_machines"]
        vms: List[VirtualMachine] = []
        for item in raw_items:
            vms.append(
                VirtualMachine(
                    vm_id=item["vm_id"],
                    vm_type=item["vm_type"],
                    host_machine_id=item["host_machine_id"],
                    capacity=ResourceCapacity(
                        cpu_mips=float(item["cpu_mips"]),
                        memory_mb=float(item["memory_mb"]),
                        bandwidth_mbps=float(item["bandwidth_mbps"]),
                    ),
                    power=PowerProfile(
                        idle_watts=float(item["idle_watts"]),
                        max_watts=float(item["max_watts"]),
                    ),
                    vcpu_count=int(item["vcpu_count"]),
                    availability_time=float(item.get("availability_time", 0.0)),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return vms

    def create_tasks(self, dataset: Dict[str, Any]) -> List[Task]:
        raw_items = dataset["tasks"]
        tasks: List[Task] = []
        for item in raw_items:
            tasks.append(
                Task(
                    task_id=item["task_id"],
                    workload_mi=float(item["workload_mi"]),
                    arrival_time=float(item["arrival_time"]),
                    deadline=float(item["deadline"]),
                    cpu_demand_mips=float(item["cpu_demand_mips"]),
                    memory_demand_mb=float(item["memory_demand_mb"]),
                    io_size_mb=float(item["io_size_mb"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return tasks

    def create_datacenter(self, dataset: Dict[str, Any]) -> Datacenter:
        dc = dataset["datacenter"]
        return Datacenter(
            datacenter_id=dc["datacenter_id"],
            physical_machines=self.create_physical_machines(dataset),
            virtual_machines=self.create_virtual_machines(dataset),
            metadata=dict(dc.get("metadata", {})),
        )

    def create_cloud_configuration(self, dataset: Dict[str, Any]) -> CloudConfiguration:
        return CloudConfiguration(
            datacenter=self.create_datacenter(dataset),
            tasks=self.create_tasks(dataset),
            epoch_length=float(dataset["epoch_length"]),
            metadata=dict(dataset.get("metadata", {})),
        )

    def invoke_scheduler(self, configuration: CloudConfiguration) -> SchedulingResult:
        return self.scheduler.schedule(
            tasks=configuration.tasks,
            virtual_machines=configuration.datacenter.virtual_machines,
        )

    def insert_events(
        self,
        simulation: Simulation,
        configuration: CloudConfiguration,
        assignment: SchedulingResult,
    ) -> None:
        tasks_by_id = {task.task_id: task for task in configuration.tasks}
        vms_by_id = {vm.vm_id: vm for vm in configuration.datacenter.virtual_machines}

        for task_id, vm_id in assignment.task_to_vm.items():
            if task_id not in tasks_by_id:
                raise ValueError(f"assignment contains unknown task_id: {task_id}")
            if vm_id not in vms_by_id:
                raise ValueError(f"assignment contains unknown vm_id: {vm_id}")

            task = tasks_by_id[task_id]
            vm = vms_by_id[vm_id]
            self._validate_assignment_feasibility(task=task, vm=vm)
            duration = self._estimate_duration(task=task, vm=vm)

            simulation.schedule(
                Event(
                    time=task.arrival_time,
                    name="vm.enqueue",
                    payload={
                        "vm_id": vm.vm_id,
                        "task": task,
                        "duration": duration,
                    },
                )
            )

    def run(self, dataset_source: Dict[str, Any] | str | Path) -> SimulationState:
        dataset = self.load_dataset(dataset_source)
        configuration = self.create_cloud_configuration(dataset)

        simulation = Simulation()
        vm_execution = VirtualMachineExecutionManager.from_virtual_machines(
            configuration.datacenter.virtual_machines
        )
        timing_metrics = TimingMetricsCollector.from_virtual_machines(
            configuration.datacenter.virtual_machines
        )
        utilization = UtilizationTracker.from_virtual_machines(
            configuration.datacenter.virtual_machines
        )
        metrics_collector = MetricsCollector(
            configuration=configuration,
            utilization_tracker=utilization,
        )
        vm_execution.register(simulation)
        timing_metrics.register(simulation)
        utilization.register(simulation)
        metrics_collector.register(simulation)

        assignment = self.invoke_scheduler(configuration)
        self.insert_events(
            simulation=simulation,
            configuration=configuration,
            assignment=assignment,
        )

        simulation.state.set("cloud_configuration", configuration)
        simulation.state.set("assignment", assignment)
        simulation.state.set("vm_execution", vm_execution)
        simulation.state.set("timing_metrics", timing_metrics)
        simulation.state.set("utilization_tracker", utilization)
        simulation.state.set("metrics_collector", metrics_collector)

        simulation.run(until=configuration.epoch_length)
        utilization.finalize(end_time=configuration.epoch_length)
        metrics = metrics_collector.finalize()
        simulation.state.set("metrics", metrics)

        runtime = OrchestratorRuntime(
            configuration=configuration,
            simulation=simulation,
            vm_execution=vm_execution,
            timing_metrics=timing_metrics,
            utilization=utilization,
            metrics=metrics,
            assignment=assignment,
        )
        simulation.state.set("orchestrator_runtime", runtime)
        return simulation.state

    @staticmethod
    def _estimate_duration(task: Task, vm: VirtualMachine) -> float:
        if vm.capacity.cpu_mips <= 0:
            raise ValueError(f"vm cpu_mips must be > 0 for duration estimate, vm_id={vm.vm_id}")
        return task.workload_mi / vm.capacity.cpu_mips

    @staticmethod
    def _validate_assignment_feasibility(task: Task, vm: VirtualMachine) -> None:
        if task.memory_demand_mb > vm.capacity.memory_mb:
            raise ValueError(
                "infeasible assignment: "
                f"task_id={task.task_id} requires memory_demand_mb={task.memory_demand_mb} "
                f"but vm_id={vm.vm_id} provides memory_mb={vm.capacity.memory_mb}"
            )
