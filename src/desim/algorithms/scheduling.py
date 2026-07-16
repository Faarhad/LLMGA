from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
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


class Scheduler(ABC):
    """Abstract scheduler that returns task-to-VM mapping only."""

    @abstractmethod
    def schedule(self, tasks: List[Task], virtual_machines: List[VirtualMachine]) -> SchedulingResult:
        raise NotImplementedError


class RandomScheduler(Scheduler):
    """Uniform random task-to-VM mapper."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def schedule(self, tasks: List[Task], virtual_machines: List[VirtualMachine]) -> SchedulingResult:
        if not virtual_machines and tasks:
            raise ValueError("cannot schedule tasks without virtual machines")

        vm_ids = [vm.vm_id for vm in virtual_machines]
        mapping: Dict[str, str] = {}
        for task in tasks:
            mapping[task.task_id] = self._rng.choice(vm_ids)
        return SchedulingResult(task_to_vm=mapping)

    def generate_random_schedules(
        self,
        tasks: List[Task],
        virtual_machines: List[VirtualMachine],
        count: int,
    ) -> List[SchedulingResult]:
        if count < 0:
            raise ValueError("count must be >= 0")
        return [self.schedule(tasks, virtual_machines) for _ in range(count)]
