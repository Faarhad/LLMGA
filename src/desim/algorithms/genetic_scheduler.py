import random
from typing import Dict, List

from ..framework.models import Task, VirtualMachine
from .base import Scheduler, SchedulingResult, VmStateView


class GeneticAlgorithmScheduler(Scheduler):
    """Pure GA scheduler that evolves task-to-VM mappings."""

    def __init__(
        self,
        population_size: int = 60,
        generations: int = 120,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.08,
        tournament_size: int = 3,
        elitism_count: int = 1,
        sla_weight: float = 2.0,
        seed: int | None = None,
    ) -> None:
        if population_size <= 1:
            raise ValueError("population_size must be > 1")
        if generations <= 0:
            raise ValueError("generations must be > 0")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if tournament_size <= 0:
            raise ValueError("tournament_size must be > 0")
        if elitism_count < 0:
            raise ValueError("elitism_count must be >= 0")
        if elitism_count >= population_size:
            raise ValueError("elitism_count must be < population_size")
        if sla_weight < 0:
            raise ValueError("sla_weight must be >= 0")

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.sla_weight = sla_weight
        self._rng = random.Random(seed)

    def schedule(self, waiting_tasks: List[Task], vm_states: List[VmStateView]) -> SchedulingResult:
        virtual_machines = [vm_state.vm for vm_state in vm_states]

        if not virtual_machines and waiting_tasks:
            raise ValueError("cannot schedule tasks without virtual machines")
        if not waiting_tasks:
            return SchedulingResult(task_to_vm={})

        feasible = self._build_feasible_vm_indices(waiting_tasks, virtual_machines)
        population = [self._random_chromosome(feasible) for _ in range(self.population_size)]

        for _ in range(self.generations):
            ranked = sorted(population, key=lambda c: self._objective(c, waiting_tasks, virtual_machines))
            next_population = ranked[: self.elitism_count]

            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(population, waiting_tasks, virtual_machines)
                parent_b = self._tournament_select(population, waiting_tasks, virtual_machines)

                child_a, child_b = self._crossover(parent_a, parent_b)
                self._mutate(child_a, feasible)
                self._mutate(child_b, feasible)

                next_population.append(child_a)
                if len(next_population) < self.population_size:
                    next_population.append(child_b)

            population = next_population

        best = min(population, key=lambda c: self._objective(c, waiting_tasks, virtual_machines))
        mapping = {
            task.task_id: virtual_machines[best[i]].vm_id
            for i, task in enumerate(waiting_tasks)
        }
        return SchedulingResult(task_to_vm=mapping)

    def _build_feasible_vm_indices(
        self,
        tasks: List[Task],
        virtual_machines: List[VirtualMachine],
    ) -> List[List[int]]:
        feasible: List[List[int]] = []
        for task in tasks:
            candidates: List[int] = []
            for i, vm in enumerate(virtual_machines):
                if task.memory_demand_mb <= vm.capacity.memory_mb and task.cpu_demand_mips <= vm.capacity.cpu_mips:
                    candidates.append(i)

            if not candidates:
                raise ValueError(
                    "no feasible VM for task "
                    f"task_id={task.task_id} memory_demand_mb={task.memory_demand_mb} "
                    f"cpu_demand_mips={task.cpu_demand_mips}"
                )
            feasible.append(candidates)
        return feasible

    def _random_chromosome(self, feasible_vm_indices: List[List[int]]) -> List[int]:
        return [self._rng.choice(candidates) for candidates in feasible_vm_indices]

    def _tournament_select(
        self,
        population: List[List[int]],
        tasks: List[Task],
        virtual_machines: List[VirtualMachine],
    ) -> List[int]:
        sample_size = min(self.tournament_size, len(population))
        candidates = self._rng.sample(population, sample_size)
        winner = min(candidates, key=lambda c: self._objective(c, tasks, virtual_machines))
        return list(winner)

    def _crossover(self, parent_a: List[int], parent_b: List[int]) -> tuple[List[int], List[int]]:
        if len(parent_a) != len(parent_b):
            raise ValueError("parents must have same chromosome length")
        if len(parent_a) <= 1 or self._rng.random() >= self.crossover_rate:
            return list(parent_a), list(parent_b)

        point = self._rng.randint(1, len(parent_a) - 1)
        child_a = parent_a[:point] + parent_b[point:]
        child_b = parent_b[:point] + parent_a[point:]
        return child_a, child_b

    def _mutate(self, chromosome: List[int], feasible_vm_indices: List[List[int]]) -> None:
        for gene_index, candidates in enumerate(feasible_vm_indices):
            if self._rng.random() < self.mutation_rate:
                chromosome[gene_index] = self._rng.choice(candidates)

    def _objective(
        self,
        chromosome: List[int],
        tasks: List[Task],
        virtual_machines: List[VirtualMachine],
    ) -> float:
        vm_ready: Dict[int, float] = {
            i: virtual_machines[i].availability_time for i in range(len(virtual_machines))
        }

        makespan = 0.0
        total_normalized_violation = 0.0

        for task_index, task in sorted(enumerate(tasks), key=lambda p: (p[1].arrival_time, p[0])):
            vm_index = chromosome[task_index]
            vm = virtual_machines[vm_index]

            start_time = max(task.arrival_time, vm_ready[vm_index])
            duration = task.workload_mi / vm.capacity.cpu_mips
            finish_time = start_time + duration
            vm_ready[vm_index] = finish_time

            makespan = max(makespan, finish_time)
            violation = max(0.0, finish_time - task.deadline)
            deadline = task.deadline if task.deadline > 0 else 1e-9
            total_normalized_violation += violation / deadline

        average_violation = total_normalized_violation / len(tasks)
        return makespan + (self.sla_weight * average_violation)


def create_scheduler(options: dict | None = None, seed: int | None = None) -> GeneticAlgorithmScheduler:
    options = options or {}
    selected_seed = options.get("seed", seed)
    return GeneticAlgorithmScheduler(
        population_size=int(options.get("population_size", 60)),
        generations=int(options.get("generations", 120)),
        crossover_rate=float(options.get("crossover_rate", 0.9)),
        mutation_rate=float(options.get("mutation_rate", 0.08)),
        tournament_size=int(options.get("tournament_size", 3)),
        elitism_count=int(options.get("elitism_count", 1)),
        sla_weight=float(options.get("sla_weight", 2.0)),
        seed=selected_seed,
    )
