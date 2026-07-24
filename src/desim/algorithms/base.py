from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from ..framework.models import Task, VirtualMachine


@dataclass(frozen=True)
class SchedulingResult:
    """Scheduler output: only task-to-VM mapping."""

    task_to_vm: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for task_id, vm_id in self.task_to_vm.items():
            if not task_id:
                raise ValueError("task ids in mapping must be non-empty")
            if not vm_id:
                raise ValueError("vm ids in mapping must be non-empty")


@dataclass(frozen=True)
class VmQueuedTaskView:
    """Read-only view of a queued VM task for scheduler decisions."""

    task: Task
    remaining_duration: float


@dataclass(frozen=True)
class VmRunningTaskView:
    """Read-only view of the currently running VM task."""

    task: Task
    remaining_duration: float


@dataclass(frozen=True)
class VmStateView:
    """Read-only VM execution snapshot exposed to schedulers."""

    vm: VirtualMachine
    availability_time: float
    queue: List[VmQueuedTaskView] = field(default_factory=list)
    running_task: VmRunningTaskView | None = None


class Scheduler(ABC):
    """Abstract scheduler that returns task-to-VM mapping only."""

    @abstractmethod
    def schedule(self, waiting_tasks: List[Task], vm_states: List[VmStateView]) -> SchedulingResult:
        raise NotImplementedError
