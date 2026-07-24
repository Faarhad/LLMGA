from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..algorithms.scheduling import Scheduler, SchedulingResult, VmQueuedTaskView, VmRunningTaskView, VmStateView
from ..research.assignment_evaluator import AssignmentEvaluation
from ..research.energy import (
    CornerPointCalibrationProvider,
    FixedCoefficientProvider,
    QuadraticEnergyModel,
    VmPowerCoefficients,
)
from ..research.fairness import FairnessModel, FairnessParameters
from ..research.metrics_collector import MetricsCollector, MetricsSnapshot
from ..research.metrics_collector import FitnessParameters
from ..research.random_benchmark import RandomBenchmarkNormalization, RandomBenchmarkNormalizer
from ..research.sla import ExponentialSLAPenaltyModel, SLAParameters
from .configuration import AppConfig
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
from .simulation import Simulation
from .state import SimulationState
from .timing_metrics import TimingMetricsCollector
from .utilization import UtilizationTracker
from .vm_execution import VirtualMachineExecutionManager


logger = logging.getLogger(__name__)


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

    def invoke_scheduler(self, waiting_tasks: List[Task], vm_states: List[VmStateView]) -> SchedulingResult:
        return self.scheduler.schedule(
            waiting_tasks=waiting_tasks,
            vm_states=vm_states,
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

    def run(
        self,
        dataset_source: Dict[str, Any] | str | Path,
        app_config: AppConfig | None = None,
        use_random_benchmark: bool = True,
    ) -> SimulationState:
        dataset = self.load_dataset(dataset_source)
        configuration = self.create_cloud_configuration(dataset)
        slot_length = self._resolve_slot_length(dataset)
        normalization = self._resolve_dynamic_normalization(
            dataset_source=dataset,
            app_config=app_config,
            use_random_benchmark=use_random_benchmark,
        )

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
        metrics_collector = self._build_metrics_collector(
            configuration=configuration,
            utilization_tracker=utilization,
            app_config=app_config,
            normalization=normalization,
        )
        self._configure_scheduler_objective(
            configuration=configuration,
            metrics_collector=metrics_collector,
        )
        vm_execution.register(simulation)
        timing_metrics.register(simulation)
        utilization.register(simulation)
        metrics_collector.register(simulation)

        simulation.dispatcher.register("task.arrival", self._on_task_arrival)
        simulation.dispatcher.register(
            "scheduler.tick",
            lambda state, event: self._on_scheduler_tick(
                simulation=simulation,
                state=state,
                event=event,
            ),
        )

        cumulative_assignment: Dict[str, str] = {}
        simulation.state.set("waiting_tasks", [])
        simulation.state.set("all_tasks_by_id", {task.task_id: task for task in configuration.tasks})
        simulation.state.set("slot_length", slot_length)

        for task in sorted(configuration.tasks, key=lambda t: (t.arrival_time, t.task_id)):
            simulation.schedule(
                Event(
                    time=task.arrival_time,
                    name="task.arrival",
                    payload={"task": task},
                )
            )

        tick_time = slot_length
        while tick_time <= configuration.epoch_length:
            simulation.schedule(
                Event(
                    time=tick_time,
                    name="scheduler.tick",
                    payload={},
                )
            )
            tick_time += slot_length

        simulation.state.set("cloud_configuration", configuration)
        simulation.state.set("assignment", SchedulingResult(task_to_vm={}))
        simulation.state.set("assignment_map", cumulative_assignment)
        simulation.state.set("vm_execution", vm_execution)
        simulation.state.set("timing_metrics", timing_metrics)
        simulation.state.set("utilization_tracker", utilization)
        simulation.state.set("metrics_collector", metrics_collector)
        simulation.state.set(
            "debug_fitness_eval",
            bool((app_config.scheduler.options if app_config is not None else {}).get("debug_fitness_evaluation", False)),
        )
        simulation.state.set("debug_fitness_predictions", [])
        if normalization is not None:
            simulation.state.set("normalization", normalization)

        simulation.run(until=configuration.epoch_length)
        self._flush_prediction_debug_logs(state=simulation.state, force=True)
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
            assignment=SchedulingResult(task_to_vm=dict(cumulative_assignment)),
        )
        simulation.state.set("assignment", runtime.assignment)
        simulation.state.set("orchestrator_runtime", runtime)
        return simulation.state

    def _build_metrics_collector(
        self,
        configuration: CloudConfiguration,
        utilization_tracker: UtilizationTracker,
        app_config: AppConfig | None,
        normalization: RandomBenchmarkNormalization | None,
    ) -> MetricsCollector:
        if app_config is None:
            return MetricsCollector(
                configuration=configuration,
                utilization_tracker=utilization_tracker,
            )

        energy_model = self._build_energy_model(app_config)
        sla_model = ExponentialSLAPenaltyModel(
            SLAParameters(
                lambda_=app_config.metrics.sla_lambda,
                theta=app_config.metrics.sla_theta,
                eta_max=app_config.metrics.sla_eta_max,
            )
        )
        fairness_model = FairnessModel(
            FairnessParameters(
                omega_1=app_config.metrics.fairness_omega_1,
                omega_2=app_config.metrics.fairness_omega_2,
                mu=app_config.metrics.fairness_mu,
            )
        )
        fitness_parameters = FitnessParameters(
            w_energy=app_config.metrics.fitness_w_energy,
            w_sla=app_config.metrics.fitness_w_sla,
            xi=app_config.metrics.fitness_xi,
            energy_norm_max=normalization.energy_norm_max if normalization is not None else 1.0,
            sla_norm_max=normalization.sla_norm_max if normalization is not None else 1.0,
        )

        return MetricsCollector(
            configuration=configuration,
            utilization_tracker=utilization_tracker,
            energy_model=energy_model,
            sla_model=sla_model,
            fairness_model=fairness_model,
            fitness_parameters=fitness_parameters,
        )

    def _resolve_dynamic_normalization(
        self,
        dataset_source: Dict[str, Any] | str | Path,
        app_config: AppConfig | None,
        use_random_benchmark: bool,
    ) -> RandomBenchmarkNormalization | None:
        if app_config is None or not use_random_benchmark:
            return None
        if not app_config.random_benchmark.enabled:
            return None

        benchmark_seed = app_config.random_seeds.global_seed
        normalizer = RandomBenchmarkNormalizer(
            dataset_loader=self.dataset_loader,
            random_seed=benchmark_seed if benchmark_seed is not None else 0,
        )
        return normalizer.compute(
            dataset_source=dataset_source,
            app_config=app_config,
            sample_count=app_config.random_benchmark.sample_count,
        )

    @staticmethod
    def _build_energy_model(app_config: AppConfig) -> QuadraticEnergyModel:
        raw_coefficients = app_config.energy.coefficients
        if raw_coefficients:
            coefficients: Dict[str, VmPowerCoefficients] = {}
            for vm_id, values in raw_coefficients.items():
                if "alpha" not in values or "beta" not in values:
                    raise ValueError(
                        f"energy.coefficients.{vm_id} must include numeric alpha and beta"
                    )
                coefficients[vm_id] = VmPowerCoefficients(
                    alpha=float(values["alpha"]),
                    beta=float(values["beta"]),
                )
            provider = FixedCoefficientProvider(coefficients=coefficients)
        else:
            provider = CornerPointCalibrationProvider(alpha_share=app_config.energy.alpha_share)

        return QuadraticEnergyModel(coefficient_provider=provider)

    def _on_task_arrival(self, state: SimulationState, event: Event) -> None:
        waiting_tasks = list(state.get("waiting_tasks", []))
        waiting_tasks.append(event.payload["task"])
        state.set("waiting_tasks", waiting_tasks)

    def _on_scheduler_tick(self, simulation: Simulation, state: SimulationState, event: Event) -> None:
        self._flush_prediction_debug_logs(state=state, force=False)

        waiting_tasks: List[Task] = list(state.get("waiting_tasks", []))
        if not waiting_tasks:
            return

        vm_execution: VirtualMachineExecutionManager = state.get("vm_execution")
        vm_states = self._build_vm_state_view(vm_execution=vm_execution, now=event.time)
        self._set_scheduler_runtime_context(now=event.time)
        assignment = self.invoke_scheduler(waiting_tasks=waiting_tasks, vm_states=vm_states)
        predicted = self._capture_scheduler_prediction()
        assignment_map: Dict[str, str] = state.get("assignment_map", {})

        waiting_by_id = {task.task_id: task for task in waiting_tasks}
        vms_by_id = {vm_state.vm.vm_id: vm_state.vm for vm_state in vm_states}
        assigned_ids: set[str] = set()

        for task_id, vm_id in assignment.task_to_vm.items():
            if task_id not in waiting_by_id:
                raise ValueError(f"scheduler assigned unknown or non-waiting task_id: {task_id}")
            if task_id in assignment_map:
                raise ValueError(f"task_id already assigned previously: {task_id}")
            if vm_id not in vms_by_id:
                raise ValueError(f"scheduler assigned unknown vm_id: {vm_id}")

            task = waiting_by_id[task_id]
            vm = vms_by_id[vm_id]
            self._validate_assignment_feasibility(task=task, vm=vm)
            duration = self._estimate_duration(task=task, vm=vm)

            simulation.schedule(
                Event(
                    time=event.time,
                    name="vm.enqueue",
                    payload={
                        "vm_id": vm.vm_id,
                        "task": task,
                        "duration": duration,
                    },
                )
            )
            assignment_map[task_id] = vm_id
            assigned_ids.add(task_id)

        state.set("assignment_map", assignment_map)
        state.set("assignment", SchedulingResult(task_to_vm=dict(assignment_map)))
        state.set("waiting_tasks", [task for task in waiting_tasks if task.task_id not in assigned_ids])

        if predicted is not None:
            self._record_prediction_debug_if_enabled(state=state, event=event, prediction=predicted)

    def _configure_scheduler_objective(
        self,
        configuration: CloudConfiguration,
        metrics_collector: MetricsCollector,
    ) -> None:
        configure = getattr(self.scheduler, "configure_paper_objective", None)
        if callable(configure):
            configure(
                datacenter=configuration.datacenter,
                fitness_parameters=metrics_collector.fitness_parameters,
                energy_model=metrics_collector.energy_model,
                sla_model=metrics_collector.sla_model,
                fairness_model=metrics_collector.fairness_model,
            )

    def _set_scheduler_runtime_context(self, now: float) -> None:
        setter = getattr(self.scheduler, "set_runtime_context", None)
        if callable(setter):
            setter(now=now)

    def _capture_scheduler_prediction(self) -> AssignmentEvaluation | None:
        getter = getattr(self.scheduler, "get_last_selected_evaluation", None)
        if callable(getter):
            return getter()
        return None

    def _record_prediction_debug_if_enabled(
        self,
        state: SimulationState,
        event: Event,
        prediction: AssignmentEvaluation,
    ) -> None:
        if not bool(state.get("debug_fitness_eval", False)):
            return

        pending: List[Dict[str, Any]] = list(state.get("debug_fitness_predictions", []))
        pending.append(
            {
                "slot_time": event.time,
                "prediction": prediction,
            }
        )
        state.set("debug_fitness_predictions", pending)

    def _flush_prediction_debug_logs(self, state: SimulationState, force: bool) -> None:
        if not bool(state.get("debug_fitness_eval", False)):
            return

        pending: List[Dict[str, Any]] = list(state.get("debug_fitness_predictions", []))
        if not pending:
            return

        metrics_collector: MetricsCollector = state.get("metrics_collector")
        utilization: UtilizationTracker = state.get("utilization_tracker")
        vm_execution: VirtualMachineExecutionManager = state.get("vm_execution")
        configuration: CloudConfiguration = state.get("cloud_configuration")

        completed_times: Dict[str, float] = dict(metrics_collector._completion_times)
        unresolved: List[Dict[str, Any]] = []

        for item in pending:
            prediction: AssignmentEvaluation = item["prediction"]
            predicted_ids = set(prediction.completion_times.keys())

            if not force and not predicted_ids.issubset(completed_times.keys()):
                unresolved.append(item)
                continue

            actual_completion_times: Dict[str, float] = {}
            for task_id in predicted_ids:
                if task_id in completed_times:
                    actual_completion_times[task_id] = completed_times[task_id]
                else:
                    actual_completion_times[task_id] = configuration.epoch_length

            actual_makespan = max(actual_completion_times.values()) if actual_completion_times else item["slot_time"]

            objective_tasks: List[Task] = []
            seen: set[str] = set()
            for vm_plan in prediction.execution_plan.values():
                for execution in vm_plan:
                    if execution.task.task_id not in seen:
                        objective_tasks.append(execution.task)
                        seen.add(execution.task.task_id)

            actual_sla = metrics_collector.sla_model.evaluate(
                tasks=objective_tasks,
                completion_times=actual_completion_times,
            )
            actual_fairness = metrics_collector.fairness_model.evaluate(actual_sla)

            compare_end = min(prediction.horizon_end, configuration.epoch_length)
            actual_energy_total = None
            if compare_end > prediction.horizon_start:
                window_energy = self._compute_window_energy(
                    configuration=configuration,
                    utilization=utilization,
                    metrics_collector=metrics_collector,
                    window_start=prediction.horizon_start,
                    window_end=compare_end,
                )
                actual_energy_total = window_energy.total_energy

            predicted_order = {
                vm_id: [execution.task.task_id for execution in vm_plan]
                for vm_id, vm_plan in prediction.execution_plan.items()
            }
            actual_order = {
                vm_id: [
                    record.task.task_id
                    for record in vm_state.task_history
                    if record.task.task_id in predicted_ids
                ]
                for vm_id, vm_state in vm_execution.vm_states.items()
            }

            logger.info(
                "ga_prediction_debug",
                extra={
                    "slot_time": item["slot_time"],
                    "predicted_makespan": prediction.makespan,
                    "actual_makespan": actual_makespan,
                    "predicted_energy": prediction.energy.total_energy,
                    "actual_energy": actual_energy_total,
                    "predicted_sla": prediction.sla.aggregate_penalty,
                    "actual_sla": actual_sla.aggregate_penalty,
                    "predicted_fairness": prediction.fairness.combined_fairness,
                    "actual_fairness": actual_fairness.combined_fairness,
                    "predicted_order": predicted_order,
                    "actual_order": actual_order,
                },
            )

        state.set("debug_fitness_predictions", unresolved)

    @staticmethod
    def _compute_window_energy(
        configuration: CloudConfiguration,
        utilization: UtilizationTracker,
        metrics_collector: MetricsCollector,
        window_start: float,
        window_end: float,
    ):
        from ..research.assignment_evaluator import AssignmentEvaluator

        helper = AssignmentEvaluator(
            datacenter=configuration.datacenter,
            energy_model=metrics_collector.energy_model,
            sla_model=metrics_collector.sla_model,
            fairness_model=metrics_collector.fairness_model,
            fitness_parameters=metrics_collector.fitness_parameters,
        )
        return helper.compute_actual_window_energy(
            utilization_snapshot=utilization.snapshot(),
            window_start=window_start,
            window_end=window_end,
        )

    @staticmethod
    def _build_vm_state_view(vm_execution: VirtualMachineExecutionManager, now: float) -> List[VmStateView]:
        vm_states: List[VmStateView] = []
        for vm_id in sorted(vm_execution.vm_states):
            vm_state = vm_execution.vm_states[vm_id]
            queue_view = [
                VmQueuedTaskView(
                    task=queued.task,
                    remaining_duration=queued.duration,
                )
                for queued in vm_state.queue
            ]
            running_view = None
            if vm_state.current_task is not None:
                running_view = VmRunningTaskView(
                    task=vm_state.current_task.task,
                    remaining_duration=max(0.0, vm_state.current_task.finish_time - now),
                )

            vm_states.append(
                VmStateView(
                    vm=vm_state.vm,
                    availability_time=vm_state.availability_time,
                    queue=queue_view,
                    running_task=running_view,
                )
            )
        return vm_states

    @staticmethod
    def _resolve_slot_length(dataset: Dict[str, Any]) -> float:
        metadata = dataset.get("metadata", {})
        candidate = dataset.get("slot_length")
        if candidate is None and isinstance(metadata, dict):
            candidate = metadata.get("slot_length")

        slot_length = float(candidate) if candidate is not None else 1.0
        if slot_length <= 0:
            raise ValueError("slot_length must be > 0")
        return slot_length

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
