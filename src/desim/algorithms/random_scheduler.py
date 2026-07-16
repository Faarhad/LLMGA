import random
from typing import Dict, List

from ..framework.models import Task, VirtualMachine
from .base import Scheduler, SchedulingResult


class RandomScheduler(Scheduler):
    """Uniform random task-to-VM mapper."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def schedule(self, tasks: List[Task], virtual_machines: List[VirtualMachine]) -> SchedulingResult:
        if not virtual_machines and tasks:
            raise ValueError("cannot schedule tasks without virtual machines")

        mapping: Dict[str, str] = {}
        for task in tasks:
            feasible_vm_ids = [
                vm.vm_id
                for vm in virtual_machines
                if task.memory_demand_mb <= vm.capacity.memory_mb and task.cpu_demand_mips <= vm.capacity.cpu_mips
            ]
            if not feasible_vm_ids:
                raise ValueError(
                    "no feasible VM for task "
                    f"task_id={task.task_id} memory_demand_mb={task.memory_demand_mb} "
                    f"cpu_demand_mips={task.cpu_demand_mips}"
                )
            mapping[task.task_id] = self._rng.choice(feasible_vm_ids)
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


def create_scheduler(options: dict | None = None, seed: int | None = None) -> RandomScheduler:
    options = options or {}
    selected_seed = options.get("seed", seed)
    return RandomScheduler(seed=selected_seed)
