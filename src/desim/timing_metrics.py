from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .event import Event
from .models import VirtualMachine
from .simulation import Simulation
from .state import SimulationState


@dataclass(frozen=True)
class TaskTimingMetrics:
    task_id: str
    vm_id: str
    arrival_time: float
    enqueue_time: float
    start_time: float
    finish_time: float
    waiting_time: float
    execution_time: float
    completion_time: float
    response_time: float
    queue_delay: float
    carry_over: float


@dataclass
class VmTimingMetrics:
    vm_id: str
    initial_availability: float
    busy_intervals: List[Tuple[float, float]] = field(default_factory=list)
    idle_intervals: List[Tuple[float, float]] = field(default_factory=list)
    availability_updates: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class TimingMetricsSnapshot:
    task_metrics: Dict[str, TaskTimingMetrics]
    vm_metrics: Dict[str, VmTimingMetrics]
    makespan: float


@dataclass
class TimingMetricsCollector:
    """Collects timing metrics using task lifecycle simulation events only."""

    vm_metrics: Dict[str, VmTimingMetrics]
    _last_finish_by_vm: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_virtual_machines(cls, virtual_machines: List[VirtualMachine]) -> "TimingMetricsCollector":
        vm_metrics: Dict[str, VmTimingMetrics] = {}
        last_finish_by_vm: Dict[str, float] = {}
        for vm in virtual_machines:
            vm_metrics[vm.vm_id] = VmTimingMetrics(
                vm_id=vm.vm_id,
                initial_availability=vm.availability_time,
                availability_updates=[(0.0, vm.availability_time)],
            )
            last_finish_by_vm[vm.vm_id] = vm.availability_time
        return cls(vm_metrics=vm_metrics, _last_finish_by_vm=last_finish_by_vm)

    def register(self, simulation: Simulation, state_key: str = "timing_metrics") -> None:
        simulation.state.set(state_key, self)
        simulation.state.set("task_timing_metrics", {})
        simulation.state.set("makespan", 0.0)
        simulation.dispatcher.register("vm.task_started", self._on_task_started)
        simulation.dispatcher.register("vm.task_finished", self._on_task_finished)

    def _on_task_started(self, state: SimulationState, event: Event) -> None:
        vm_id = event.payload["vm_id"]
        start_time = float(event.payload["start_time"])
        expected_finish = float(event.payload["expected_finish_time"])

        vm = self.vm_metrics[vm_id]
        last_finish = self._last_finish_by_vm[vm_id]
        if start_time > last_finish:
            vm.idle_intervals.append((last_finish, start_time))

        vm.availability_updates.append((start_time, expected_finish))

    def _on_task_finished(self, state: SimulationState, event: Event) -> None:
        vm_id = event.payload["vm_id"]
        task = event.payload["task"]

        arrival_time = float(event.payload["arrival_time"])
        enqueue_time = float(event.payload["enqueued_at"])
        start_time = float(event.payload["start_time"])
        finish_time = float(event.payload["finish_time"])
        initial_vm_availability = float(event.payload["initial_vm_availability"])

        waiting_time = start_time - arrival_time
        execution_time = finish_time - start_time
        completion_time = finish_time
        response_time = finish_time - arrival_time
        queue_delay = start_time - enqueue_time
        carry_over = max(0.0, min(start_time, initial_vm_availability) - arrival_time)

        task_metrics = state.get("task_timing_metrics", {})
        task_metrics[task.task_id] = TaskTimingMetrics(
            task_id=task.task_id,
            vm_id=vm_id,
            arrival_time=arrival_time,
            enqueue_time=enqueue_time,
            start_time=start_time,
            finish_time=finish_time,
            waiting_time=waiting_time,
            execution_time=execution_time,
            completion_time=completion_time,
            response_time=response_time,
            queue_delay=queue_delay,
            carry_over=carry_over,
        )
        state.set("task_timing_metrics", task_metrics)

        vm = self.vm_metrics[vm_id]
        vm.busy_intervals.append((start_time, finish_time))
        vm.availability_updates.append((finish_time, finish_time))
        self._last_finish_by_vm[vm_id] = finish_time

        current_makespan = state.get("makespan", 0.0)
        if finish_time > current_makespan:
            state.set("makespan", finish_time)

    def snapshot(self, state: SimulationState) -> TimingMetricsSnapshot:
        task_metrics = dict(state.get("task_timing_metrics", {}))
        makespan = float(state.get("makespan", 0.0))
        return TimingMetricsSnapshot(
            task_metrics=task_metrics,
            vm_metrics=self.vm_metrics,
            makespan=makespan,
        )
