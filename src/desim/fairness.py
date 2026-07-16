from dataclasses import dataclass
import math
from typing import Dict

from .sla import SLAAggregateResult, TaskSLAMetrics


@dataclass(frozen=True)
class FairnessParameters:
    omega_1: float
    omega_2: float
    mu: float

    def __post_init__(self) -> None:
        if self.omega_1 < 0 or self.omega_2 < 0:
            raise ValueError("omega_1 and omega_2 must be >= 0")
        if abs((self.omega_1 + self.omega_2) - 1.0) > 1e-9:
            raise ValueError("omega_1 + omega_2 must equal 1")
        if self.mu < 0:
            raise ValueError("mu must be >= 0")


@dataclass(frozen=True)
class FairnessResult:
    penalties: Dict[str, float]
    jain_index: float
    exponential_variance: float
    combined_fairness: float


class FairnessModel:
    """Computes fairness metrics from SLA per-task penalties."""

    def __init__(self, parameters: FairnessParameters) -> None:
        self.parameters = parameters

    def evaluate(self, sla_result: SLAAggregateResult) -> FairnessResult:
        penalties = self._extract_penalties(sla_result.per_task)
        jain = self.jain_index(penalties)
        exp_var = self.exponential_variance(penalties)
        combined = self.combined_fairness(jain, exp_var)
        return FairnessResult(
            penalties=penalties,
            jain_index=jain,
            exponential_variance=exp_var,
            combined_fairness=combined,
        )

    @staticmethod
    def _extract_penalties(per_task: Dict[str, TaskSLAMetrics]) -> Dict[str, float]:
        return {task_id: metrics.penalty for task_id, metrics in per_task.items()}

    @staticmethod
    def jain_index(penalties: Dict[str, float]) -> float:
        if not penalties:
            return 1.0

        values = list(penalties.values())
        s1 = sum(values)
        s2 = sum(v * v for v in values)

        if s1 == 0.0 and s2 == 0.0:
            return 1.0

        n = len(values)
        return (s1 * s1) / (n * s2)

    def exponential_variance(self, penalties: Dict[str, float]) -> float:
        if not penalties:
            return 0.0

        values = list(penalties.values())
        n = len(values)
        mean = sum(values) / n

        total = 0.0
        for value in values:
            positive_part = max(0.0, value - mean)
            total += math.exp(self.parameters.mu * positive_part)

        return (total / n) - 1.0

    def combined_fairness(self, jain_index: float, exponential_variance: float) -> float:
        p = self.parameters
        return p.omega_1 * (1.0 - jain_index) + p.omega_2 * exponential_variance
