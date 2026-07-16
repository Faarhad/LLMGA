import math
import unittest

from desim.fairness import FairnessModel, FairnessParameters
from desim.sla import SLAAggregateResult, TaskSLAMetrics


def make_sla_result(penalties: dict[str, float]) -> SLAAggregateResult:
    per_task = {}
    for i, (task_id, penalty) in enumerate(penalties.items(), start=1):
        per_task[task_id] = TaskSLAMetrics(
            task_id=task_id,
            deadline=10.0,
            completion_time=10.0 + i,
            deadline_violation=float(i),
            normalized_violation=float(i) / 10.0,
            penalty=penalty,
        )
    return SLAAggregateResult(per_task=per_task, aggregate_penalty=sum(penalties.values()))


class TestFairnessModel(unittest.TestCase):
    def test_jain_index_equal_penalties(self) -> None:
        model = FairnessModel(FairnessParameters(omega_1=0.5, omega_2=0.5, mu=1.0))
        result = model.evaluate(make_sla_result({"t1": 2.0, "t2": 2.0, "t3": 2.0}))
        self.assertAlmostEqual(result.jain_index, 1.0)

    def test_jain_index_skewed_penalties(self) -> None:
        model = FairnessModel(FairnessParameters(omega_1=0.5, omega_2=0.5, mu=1.0))
        result = model.evaluate(make_sla_result({"t1": 6.0, "t2": 0.0, "t3": 0.0}))
        self.assertAlmostEqual(result.jain_index, 1.0 / 3.0)

    def test_exponential_variance(self) -> None:
        params = FairnessParameters(omega_1=0.5, omega_2=0.5, mu=1.0)
        model = FairnessModel(params)
        penalties = {"t1": 1.0, "t2": 3.0}
        result = model.evaluate(make_sla_result(penalties))

        mean = 2.0
        expected = ((math.exp(max(0.0, 1.0 - mean)) + math.exp(max(0.0, 3.0 - mean))) / 2.0) - 1.0
        self.assertAlmostEqual(result.exponential_variance, expected)

    def test_combined_fairness(self) -> None:
        params = FairnessParameters(omega_1=0.7, omega_2=0.3, mu=1.0)
        model = FairnessModel(params)
        result = model.evaluate(make_sla_result({"t1": 1.0, "t2": 3.0}))

        expected = params.omega_1 * (1.0 - result.jain_index) + params.omega_2 * result.exponential_variance
        self.assertAlmostEqual(result.combined_fairness, expected)

    def test_all_zero_penalties(self) -> None:
        model = FairnessModel(FairnessParameters(omega_1=0.5, omega_2=0.5, mu=2.0))
        result = model.evaluate(make_sla_result({"t1": 0.0, "t2": 0.0}))
        self.assertAlmostEqual(result.jain_index, 1.0)
        self.assertAlmostEqual(result.exponential_variance, 0.0)

    def test_invalid_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FairnessParameters(omega_1=0.3, omega_2=0.3, mu=1.0)


if __name__ == "__main__":
    unittest.main()
