from dataclasses import dataclass, field
from typing import Dict, List

from .event import Event
from .models import VirtualMachine
from .simulation import Simulation
from .state import SimulationState


@dataclass(frozen=True)
class UtilizationInterval:
    vm_id: str
    start_time: float
    end_time: float
    utilization: float

    def __post_init__(self) -> None:
        if self.start_time < 0 or self.end_time < 0:
            raise ValueError("interval times must be >= 0")
        if self.end_time < self.start_time:
            raise ValueError("interval end_time must be >= start_time")
        if self.utilization < 0:
            raise ValueError("utilization must be >= 0")


@dataclass
class VmUtilizationTrace:
    vm_id: str
    current_utilization: float
    last_change_time: float
    intervals: List[UtilizationInterval] = field(default_factory=list)


@dataclass
class UtilizationSnapshot:
    traces: Dict[str, VmUtilizationTrace]


@dataclass
class UtilizationTracker:
    """Event-driven utilization tracking with piecewise-constant intervals."""

    traces: Dict[str, VmUtilizationTrace]
    _vm_cpu_mips: Dict[str, float]

    @classmethod
    def from_virtual_machines(cls, virtual_machines: List[VirtualMachine]) -> "UtilizationTracker":
        traces: Dict[str, VmUtilizationTrace] = {}
        vm_cpu_mips: Dict[str, float] = {}

        for vm in virtual_machines:
            traces[vm.vm_id] = VmUtilizationTrace(
                vm_id=vm.vm_id,
                current_utilization=0.0,
                last_change_time=0.0,
            )
            vm_cpu_mips[vm.vm_id] = vm.capacity.cpu_mips

        return cls(traces=traces, _vm_cpu_mips=vm_cpu_mips)

    def register(self, simulation: Simulation, state_key: str = "utilization_tracker") -> None:
        simulation.state.set(state_key, self)
        simulation.dispatcher.register("vm.task_started", self._on_task_started)
        simulation.dispatcher.register("vm.task_finished", self._on_task_finished)

    def _on_task_started(self, state: SimulationState, event: Event) -> None:
        vm_id = event.payload["vm_id"]
        start_time = float(event.payload["start_time"])
        task = event.payload["task"]

        trace = self._get_trace(vm_id)
        self._append_interval_if_needed(trace=trace, interval_end=start_time)

        vm_cpu = self._vm_cpu_mips[vm_id]
        if vm_cpu <= 0:
            raise ValueError(f"vm cpu_mips must be > 0 for utilization, vm_id={vm_id}")

        new_utilization = task.cpu_demand_mips / vm_cpu
        if new_utilization < 0:
            raise ValueError("utilization cannot be negative")

        trace.current_utilization = new_utilization
        trace.last_change_time = start_time

    def _on_task_finished(self, state: SimulationState, event: Event) -> None:
        vm_id = event.payload["vm_id"]
        finish_time = float(event.payload["finish_time"])

        trace = self._get_trace(vm_id)
        self._append_interval_if_needed(trace=trace, interval_end=finish_time)

        trace.current_utilization = 0.0
        trace.last_change_time = finish_time

    def finalize(self, end_time: float) -> None:
        if end_time < 0:
            raise ValueError("end_time must be >= 0")
        for trace in self.traces.values():
            self._append_interval_if_needed(trace=trace, interval_end=end_time)

    def snapshot(self) -> UtilizationSnapshot:
        return UtilizationSnapshot(traces=self.traces)

    def _get_trace(self, vm_id: str) -> VmUtilizationTrace:
        if vm_id not in self.traces:
            raise KeyError(f"unknown vm_id: {vm_id}")
        return self.traces[vm_id]

    @staticmethod
    def _append_interval_if_needed(trace: VmUtilizationTrace, interval_end: float) -> None:
        if interval_end < trace.last_change_time:
            raise ValueError("interval end cannot be before last change time")
        if interval_end == trace.last_change_time:
            return
        trace.intervals.append(
            UtilizationInterval(
                vm_id=trace.vm_id,
                start_time=trace.last_change_time,
                end_time=interval_end,
                utilization=trace.current_utilization,
            )
        )
