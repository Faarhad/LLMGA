from dataclasses import dataclass
import math
from typing import Dict, Iterable

from .models import Task


@dataclass(frozen=True)
class SLAParameters:
    lambda_: float
    theta: float
    eta_max: float

    def __post_init__(self) -> None:
        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be > 0")
        if self.theta <= 0:
            raise ValueError("theta must be > 0")
        if self.eta_max <= 0:
            raise ValueError("eta_max must be > 0")

    @property
    def kappa(self) -> float:
        return self.lambda_ * self.theta * math.exp(self.theta * self.eta_max)


@dataclass(frozen=True)
class TaskSLAMetrics:
    task_id: str
    deadline: float
    completion_time: float
    deadline_violation: float
    normalized_violation: float
    penalty: float


@dataclass(frozen=True)
class SLAAggregateResult:
    per_task: Dict[str, TaskSLAMetrics]
    aggregate_penalty: float


class ExponentialSLAPenaltyModel:
    """Implements deadline violation and piecewise exponential SLA penalty."""

    def __init__(self, parameters: SLAParameters) -> None:
        self.parameters = parameters

    def evaluate(self, tasks: Iterable[Task], completion_times: Dict[str, float]) -> SLAAggregateResult:
        per_task: Dict[str, TaskSLAMetrics] = {}
        total_penalty = 0.0

        for task in tasks:
            if task.task_id not in completion_times:
                raise KeyError(f"missing completion time for task_id={task.task_id}")
            if task.deadline <= 0:
                raise ValueError(f"task deadline must be > 0 for normalization, task_id={task.task_id}")

            completion_time = float(completion_times[task.task_id])
            violation = max(0.0, completion_time - task.deadline)
            eta = violation / task.deadline
            penalty = self._penalty(eta=eta, violation=violation)

            metrics = TaskSLAMetrics(
                task_id=task.task_id,
                deadline=task.deadline,
                completion_time=completion_time,
                deadline_violation=violation,
                normalized_violation=eta,
                penalty=penalty,
            )
            per_task[task.task_id] = metrics
            total_penalty += penalty

        return SLAAggregateResult(per_task=per_task, aggregate_penalty=total_penalty)

    def _penalty(self, eta: float, violation: float) -> float:
        p = self.parameters

        if violation == 0.0:
            return 0.0

        if 0.0 < eta <= p.eta_max:
            return p.lambda_ * (math.exp(p.theta * eta) - 1.0)

        return p.lambda_ * (math.exp(p.theta * p.eta_max) - 1.0) + p.kappa * (eta - p.eta_max)
