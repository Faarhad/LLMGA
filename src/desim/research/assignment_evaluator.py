from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..algorithms.base import VmStateView
from ..framework.models import Datacenter, Task
from ..framework.utilization import UtilizationInterval, UtilizationSnapshot, VmUtilizationTrace
from .energy import DatacenterEnergyBreakdown, QuadraticEnergyModel
from .fairness import FairnessModel, FairnessResult
from .metrics_collector import FitnessParameters
from .paper_objective import compute_paper_fitness, compute_sla_objective
from .sla import ExponentialSLAPenaltyModel, SLAAggregateResult


@dataclass(frozen=True)
class PredictedTaskExecution:
    task: Task
    vm_id: str
    start_time: float
    finish_time: float


@dataclass(frozen=True)
class AssignmentEvaluation:
    objective: float
    fitness: float
    makespan: float
    energy: DatacenterEnergyBreakdown
    sla: SLAAggregateResult
    fairness: FairnessResult
    sla_objective: float
    completion_times: Dict[str, float]
    execution_plan: Dict[str, List[PredictedTaskExecution]]
    horizon_start: float
    horizon_end: float


class AssignmentEvaluator:
    """Predicts assignment outcomes from current VM running/queue state."""

    def __init__(
        self,
        datacenter: Datacenter,
        energy_model: QuadraticEnergyModel,
        sla_model: ExponentialSLAPenaltyModel,
        fairness_model: FairnessModel,
        fitness_parameters: FitnessParameters,
    ) -> None:
        self.datacenter = datacenter
        self.energy_model = energy_model
        self.sla_model = sla_model
        self.fairness_model = fairness_model
        self.fitness_parameters = fitness_parameters

    def evaluate(
        self,
        waiting_tasks: Sequence[Task],
        vm_states: Sequence[VmStateView],
        assignment: Dict[str, str],
        now: float,
    ) -> AssignmentEvaluation:
        assigned_ordered = [task for task in waiting_tasks if task.task_id in assignment]

        vm_state_by_id = {vm_state.vm.vm_id: vm_state for vm_state in vm_states}
        execution_plan: Dict[str, List[PredictedTaskExecution]] = {
            vm_state.vm.vm_id: [] for vm_state in vm_state_by_id.values()
        }

        for task in assigned_ordered:
            vm_id = assignment[task.task_id]
            if vm_id not in vm_state_by_id:
                raise ValueError(f"assignment contains unknown vm_id: {vm_id}")
            vm = vm_state_by_id[vm_id].vm
            if task.memory_demand_mb > vm.capacity.memory_mb or task.cpu_demand_mips > vm.capacity.cpu_mips:
                raise ValueError(f"infeasible task assignment task_id={task.task_id} vm_id={vm_id}")

        assigned_by_vm: Dict[str, List[Task]] = {vm_id: [] for vm_id in vm_state_by_id}
        for task in assigned_ordered:
            assigned_by_vm[assignment[task.task_id]].append(task)

        completion_times: Dict[str, float] = {}

        for vm_id, vm_state in vm_state_by_id.items():
            vm = vm_state.vm
            vm_plan: List[PredictedTaskExecution] = []

            cursor = max(now, vm_state.availability_time)

            if vm_state.running_task is not None:
                if vm_state.running_task.remaining_duration < 0:
                    raise ValueError("running task remaining_duration must be >= 0")
                running_start = now
                running_finish = now + vm_state.running_task.remaining_duration
                vm_plan.append(
                    PredictedTaskExecution(
                        task=vm_state.running_task.task,
                        vm_id=vm_id,
                        start_time=running_start,
                        finish_time=running_finish,
                    )
                )
                completion_times[vm_state.running_task.task.task_id] = running_finish
                cursor = running_finish

            for queued in vm_state.queue:
                if queued.remaining_duration < 0:
                    raise ValueError("queued task remaining_duration must be >= 0")
                start_time = max(queued.task.arrival_time, cursor)
                finish_time = start_time + queued.remaining_duration
                vm_plan.append(
                    PredictedTaskExecution(
                        task=queued.task,
                        vm_id=vm_id,
                        start_time=start_time,
                        finish_time=finish_time,
                    )
                )
                completion_times[queued.task.task_id] = finish_time
                cursor = finish_time

            for task in assigned_by_vm[vm_id]:
                duration = self._duration(task=task, vm_cpu_mips=vm.capacity.cpu_mips)
                start_time = max(task.arrival_time, cursor)
                finish_time = start_time + duration
                vm_plan.append(
                    PredictedTaskExecution(
                        task=task,
                        vm_id=vm_id,
                        start_time=start_time,
                        finish_time=finish_time,
                    )
                )
                completion_times[task.task_id] = finish_time
                cursor = finish_time

            execution_plan[vm_id] = vm_plan

        if not completion_times:
            raise ValueError("evaluation requires at least one running, queued, or assigned task")

        makespan = max(completion_times.values())
        horizon_end = max(makespan, now + 1e-9)

        utilization_snapshot = self._build_window_utilization_snapshot(
            execution_plan=execution_plan,
            vm_states=vm_states,
            window_start=now,
            window_end=horizon_end,
        )
        energy = self.energy_model.compute_datacenter_energy(
            datacenter=self.datacenter,
            utilization_snapshot=utilization_snapshot,
            epoch_length=horizon_end - now,
        )

        objective_tasks = self._collect_objective_tasks(vm_states=vm_states, assigned_tasks=assigned_ordered)
        objective_completion = {task.task_id: completion_times[task.task_id] for task in objective_tasks}
        sla = self.sla_model.evaluate(tasks=objective_tasks, completion_times=objective_completion)
        fairness = self.fairness_model.evaluate(sla)

        sla_objective = compute_sla_objective(
            aggregate_sla_penalty=sla.aggregate_penalty,
            combined_fairness=fairness.combined_fairness,
            xi=self.fitness_parameters.xi,
        )
        fitness = compute_paper_fitness(
            energy_total=energy.total_energy,
            aggregate_sla_penalty=sla.aggregate_penalty,
            combined_fairness=fairness.combined_fairness,
            fitness_parameters=self.fitness_parameters,
        )

        return AssignmentEvaluation(
            objective=fitness,
            fitness=fitness,
            makespan=makespan,
            energy=energy,
            sla=sla,
            fairness=fairness,
            sla_objective=sla_objective,
            completion_times=completion_times,
            execution_plan=execution_plan,
            horizon_start=now,
            horizon_end=horizon_end,
        )

    def compute_actual_window_energy(
        self,
        utilization_snapshot: UtilizationSnapshot,
        window_start: float,
        window_end: float,
    ) -> DatacenterEnergyBreakdown:
        if window_end <= window_start:
            raise ValueError("window_end must be > window_start")

        clipped = self._clip_snapshot(
            utilization_snapshot=utilization_snapshot,
            window_start=window_start,
            window_end=window_end,
        )
        return self.energy_model.compute_datacenter_energy(
            datacenter=self.datacenter,
            utilization_snapshot=clipped,
            epoch_length=window_end - window_start,
        )

    def _build_window_utilization_snapshot(
        self,
        execution_plan: Dict[str, List[PredictedTaskExecution]],
        vm_states: Sequence[VmStateView],
        window_start: float,
        window_end: float,
    ) -> UtilizationSnapshot:
        traces: Dict[str, VmUtilizationTrace] = {}

        for vm_state in vm_states:
            vm = vm_state.vm
            intervals: List[UtilizationInterval] = []
            cursor = window_start

            for execution in execution_plan.get(vm.vm_id, []):
                start = max(execution.start_time, window_start)
                end = min(execution.finish_time, window_end)
                if end <= start:
                    continue

                if start > cursor:
                    intervals.append(
                        UtilizationInterval(
                            vm_id=vm.vm_id,
                            start_time=cursor - window_start,
                            end_time=start - window_start,
                            utilization=0.0,
                        )
                    )

                utilization = execution.task.cpu_demand_mips / vm.capacity.cpu_mips
                intervals.append(
                    UtilizationInterval(
                        vm_id=vm.vm_id,
                        start_time=start - window_start,
                        end_time=end - window_start,
                        utilization=utilization,
                    )
                )
                cursor = end

            if cursor < window_end:
                intervals.append(
                    UtilizationInterval(
                        vm_id=vm.vm_id,
                        start_time=cursor - window_start,
                        end_time=window_end - window_start,
                        utilization=0.0,
                    )
                )

            traces[vm.vm_id] = VmUtilizationTrace(
                vm_id=vm.vm_id,
                current_utilization=0.0,
                last_change_time=window_end - window_start,
                intervals=intervals,
            )

        return UtilizationSnapshot(traces=traces)

    def _clip_snapshot(
        self,
        utilization_snapshot: UtilizationSnapshot,
        window_start: float,
        window_end: float,
    ) -> UtilizationSnapshot:
        traces: Dict[str, VmUtilizationTrace] = {}
        for vm in self.datacenter.virtual_machines:
            trace = utilization_snapshot.traces.get(vm.vm_id)
            if trace is None:
                raise KeyError(f"missing utilization trace for vm_id={vm.vm_id}")

            clipped: List[UtilizationInterval] = []
            cursor = window_start

            for interval in trace.intervals:
                start = max(interval.start_time, window_start)
                end = min(interval.end_time, window_end)
                if end <= start:
                    continue
                if start > cursor:
                    clipped.append(
                        UtilizationInterval(
                            vm_id=vm.vm_id,
                            start_time=cursor - window_start,
                            end_time=start - window_start,
                            utilization=0.0,
                        )
                    )
                clipped.append(
                    UtilizationInterval(
                        vm_id=vm.vm_id,
                        start_time=start - window_start,
                        end_time=end - window_start,
                        utilization=interval.utilization,
                    )
                )
                cursor = end

            if cursor < window_end:
                clipped.append(
                    UtilizationInterval(
                        vm_id=vm.vm_id,
                        start_time=cursor - window_start,
                        end_time=window_end - window_start,
                        utilization=0.0,
                    )
                )

            traces[vm.vm_id] = VmUtilizationTrace(
                vm_id=vm.vm_id,
                current_utilization=0.0,
                last_change_time=window_end - window_start,
                intervals=clipped,
            )

        return UtilizationSnapshot(traces=traces)

    @staticmethod
    def _collect_objective_tasks(vm_states: Sequence[VmStateView], assigned_tasks: Sequence[Task]) -> List[Task]:
        collected: Dict[str, Task] = {}
        for vm_state in vm_states:
            if vm_state.running_task is not None:
                collected[vm_state.running_task.task.task_id] = vm_state.running_task.task
            for queued in vm_state.queue:
                collected[queued.task.task_id] = queued.task
        for task in assigned_tasks:
            collected[task.task_id] = task
        return list(collected.values())

    @staticmethod
    def _duration(task: Task, vm_cpu_mips: float) -> float:
        if vm_cpu_mips <= 0:
            raise ValueError("vm cpu_mips must be > 0")
        return task.workload_mi / vm_cpu_mips
