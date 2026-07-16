from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .event import Event
from .models import Task, VirtualMachine
from .simulation import Simulation
from .state import SimulationState


@dataclass(frozen=True)
class QueuedTask:
    """Task queued for a VM with an externally provided execution duration."""

    task: Task
    duration: float
    enqueued_at: float

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be > 0")
        if self.enqueued_at < 0:
            raise ValueError("enqueued_at must be >= 0")


@dataclass(frozen=True)
class RunningTask:
    """Current running task on a VM."""

    task: Task
    enqueued_at: float
    arrival_time: float
    start_time: float
    finish_time: float

    def __post_init__(self) -> None:
        if self.enqueued_at < 0:
            raise ValueError("enqueued_at must be >= 0")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be >= 0")
        if self.start_time < 0:
            raise ValueError("start_time must be >= 0")
        if self.finish_time < self.start_time:
            raise ValueError("finish_time must be >= start_time")


@dataclass(frozen=True)
class TaskExecutionRecord:
    """Completed task execution record."""

    task: Task
    enqueued_at: float
    arrival_time: float
    start_time: float
    finish_time: float


@dataclass
class VirtualMachineExecutionState:
    """Execution state and queue for one VM."""

    vm: VirtualMachine
    queue: Deque[QueuedTask] = field(default_factory=deque)
    current_task: Optional[RunningTask] = None
    availability_time: float = field(init=False)
    initial_availability_time: float = field(init=False)
    availability_updates: List[Tuple[float, float]] = field(default_factory=list)
    busy_intervals: List[Tuple[float, float]] = field(default_factory=list)
    idle_intervals: List[Tuple[float, float]] = field(default_factory=list)
    task_history: List[TaskExecutionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.availability_time = self.vm.availability_time
        self.initial_availability_time = self.vm.availability_time
        self.availability_updates.append((0.0, self.availability_time))

    def enqueue(self, task: Task, duration: float, enqueued_at: float) -> None:
        self.queue.append(QueuedTask(task=task, duration=duration, enqueued_at=enqueued_at))

    def dequeue(self) -> QueuedTask:
        if not self.queue:
            raise IndexError("dequeue from empty VM queue")
        return self.queue.popleft()

    def start_task(self, at_time: float) -> RunningTask:
        if at_time < 0:
            raise ValueError("at_time must be >= 0")
        if self.current_task is not None:
            raise RuntimeError("cannot start a task while another task is running")
        queued = self.dequeue()

        start_time = max(at_time, self.availability_time)
        if start_time > self.availability_time:
            self.idle_intervals.append((self.availability_time, start_time))

        finish_time = start_time + queued.duration
        running = RunningTask(
            task=queued.task,
            enqueued_at=queued.enqueued_at,
            arrival_time=queued.task.arrival_time,
            start_time=start_time,
            finish_time=finish_time,
        )
        self.current_task = running
        self.availability_time = finish_time
        self.availability_updates.append((start_time, finish_time))
        return running

    def finish_task(self, at_time: float) -> TaskExecutionRecord:
        if self.current_task is None:
            raise RuntimeError("cannot finish task when no task is running")
        if at_time < self.current_task.start_time:
            raise ValueError("finish time cannot be before start time")

        running = self.current_task
        self.current_task = None
        self.availability_time = at_time

        self.busy_intervals.append((running.start_time, at_time))
        record = TaskExecutionRecord(
            task=running.task,
            enqueued_at=running.enqueued_at,
            arrival_time=running.arrival_time,
            start_time=running.start_time,
            finish_time=at_time,
        )
        self.task_history.append(record)

        if not self.availability_updates or self.availability_updates[-1][1] != at_time:
            self.availability_updates.append((at_time, at_time))
        return record

    def queue_length(self) -> int:
        return len(self.queue)


@dataclass
class VirtualMachineExecutionManager:
    """Event-driven VM execution manager for the simulation engine."""

    vm_states: Dict[str, VirtualMachineExecutionState]
    _pending_start_events: set[str] = field(default_factory=set)

    @classmethod
    def from_virtual_machines(cls, virtual_machines: List[VirtualMachine]) -> "VirtualMachineExecutionManager":
        return cls(vm_states={vm.vm_id: VirtualMachineExecutionState(vm=vm) for vm in virtual_machines})

    def get_vm_state(self, vm_id: str) -> VirtualMachineExecutionState:
        if vm_id not in self.vm_states:
            raise KeyError(f"unknown vm_id: {vm_id}")
        return self.vm_states[vm_id]

    def register(self, simulation: Simulation, state_key: str = "vm_execution") -> None:
        simulation.state.set(state_key, self)
        simulation.dispatcher.register("vm.enqueue", lambda s, e: self._on_enqueue(simulation, s, e))
        simulation.dispatcher.register("vm.start_next", lambda s, e: self._on_start_next(simulation, s, e))
        simulation.dispatcher.register("vm.finish_current", lambda s, e: self._on_finish_current(simulation, s, e))

    def _on_enqueue(self, simulation: Simulation, state: SimulationState, event: Event) -> None:
        vm_id = event.payload["vm_id"]
        task = event.payload["task"]
        duration = event.payload["duration"]
        vm_state = self.get_vm_state(vm_id)
        vm_state.enqueue(task=task, duration=duration, enqueued_at=event.time)
        self._schedule_start_if_needed(simulation, vm_id=vm_id, now=event.time)

    def _on_start_next(self, simulation: Simulation, state: SimulationState, event: Event) -> None:
        vm_id = event.payload["vm_id"]
        self._pending_start_events.discard(vm_id)

        vm_state = self.get_vm_state(vm_id)
        if vm_state.current_task is not None or vm_state.queue_length() == 0:
            return

        running = vm_state.start_task(at_time=event.time)
        simulation.schedule(
            Event(
                time=running.start_time,
                name="vm.task_started",
                payload={
                    "vm_id": vm_id,
                    "task": running.task,
                    "enqueued_at": running.enqueued_at,
                    "arrival_time": running.arrival_time,
                    "start_time": running.start_time,
                    "expected_finish_time": running.finish_time,
                    "initial_vm_availability": vm_state.initial_availability_time,
                },
            )
        )
        simulation.schedule(
            Event(
                time=running.finish_time,
                name="vm.finish_current",
                payload={"vm_id": vm_id},
            )
        )

    def _on_finish_current(self, simulation: Simulation, state: SimulationState, event: Event) -> None:
        vm_id = event.payload["vm_id"]
        vm_state = self.get_vm_state(vm_id)
        record = vm_state.finish_task(at_time=event.time)
        simulation.schedule(
            Event(
                time=record.finish_time,
                name="vm.task_finished",
                payload={
                    "vm_id": vm_id,
                    "task": record.task,
                    "enqueued_at": record.enqueued_at,
                    "arrival_time": record.arrival_time,
                    "start_time": record.start_time,
                    "finish_time": record.finish_time,
                    "initial_vm_availability": vm_state.initial_availability_time,
                },
            )
        )

        if vm_state.queue_length() > 0:
            self._schedule_start_if_needed(simulation, vm_id=vm_id, now=event.time)

    def _schedule_start_if_needed(self, simulation: Simulation, vm_id: str, now: float) -> None:
        vm_state = self.get_vm_state(vm_id)
        if vm_state.current_task is not None or vm_state.queue_length() == 0:
            return
        if vm_id in self._pending_start_events:
            return

        start_time = max(now, vm_state.availability_time)
        self._pending_start_events.add(vm_id)
        simulation.schedule(
            Event(
                time=start_time,
                name="vm.start_next",
                payload={"vm_id": vm_id},
            )
        )
