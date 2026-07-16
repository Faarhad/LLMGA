import math
import unittest

from desim.models import Task
from desim.sla import ExponentialSLAPenaltyModel, SLAParameters


def make_task(task_id: str, deadline: float) -> Task:
    return Task(
        task_id=task_id,
        workload_mi=1000,
        arrival_time=0,
        deadline=deadline,
        cpu_demand_mips=100,
        memory_demand_mb=128,
        io_size_mb=10,
    )


class TestExponentialSLAPenaltyModel(unittest.TestCase):
    def test_no_violation_zero_penalty(self) -> None:
        model = ExponentialSLAPenaltyModel(SLAParameters(lambda_=1.0, theta=1.0, eta_max=2.0))
        t1 = make_task("t1", deadline=10.0)

        result = model.evaluate(tasks=[t1], completion_times={"t1": 10.0})

        self.assertEqual(result.per_task["t1"].deadline_violation, 0.0)
        self.assertEqual(result.per_task["t1"].normalized_violation, 0.0)
        self.assertEqual(result.per_task["t1"].penalty, 0.0)
        self.assertEqual(result.aggregate_penalty, 0.0)

    def test_exponential_regime(self) -> None:
        model = ExponentialSLAPenaltyModel(SLAParameters(lambda_=1.0, theta=1.0, eta_max=2.0))
        t1 = make_task("t1", deadline=10.0)

        result = model.evaluate(tasks=[t1], completion_times={"t1": 12.0})

        expected_eta = 0.2
        expected_penalty = math.exp(expected_eta) - 1.0

        self.assertAlmostEqual(result.per_task["t1"].deadline_violation, 2.0)
        self.assertAlmostEqual(result.per_task["t1"].normalized_violation, expected_eta)
        self.assertAlmostEqual(result.per_task["t1"].penalty, expected_penalty)

    def test_linear_tail_regime(self) -> None:
        params = SLAParameters(lambda_=1.0, theta=1.0, eta_max=2.0)
        model = ExponentialSLAPenaltyModel(params)
        t1 = make_task("t1", deadline=10.0)

        result = model.evaluate(tasks=[t1], completion_times={"t1": 40.0})

        eta = 3.0
        expected = (math.exp(2.0) - 1.0) + (params.kappa * (eta - 2.0))

        self.assertAlmostEqual(result.per_task["t1"].deadline_violation, 30.0)
        self.assertAlmostEqual(result.per_task["t1"].normalized_violation, 3.0)
        self.assertAlmostEqual(result.per_task["t1"].penalty, expected)

    def test_aggregate_penalty(self) -> None:
        model = ExponentialSLAPenaltyModel(SLAParameters(lambda_=1.0, theta=1.0, eta_max=2.0))
        t1 = make_task("t1", deadline=10.0)
        t2 = make_task("t2", deadline=10.0)

        result = model.evaluate(tasks=[t1, t2], completion_times={"t1": 10.0, "t2": 12.0})

        expected = 0.0 + (math.exp(0.2) - 1.0)
        self.assertAlmostEqual(result.aggregate_penalty, expected)

    def test_missing_completion_raises(self) -> None:
        model = ExponentialSLAPenaltyModel(SLAParameters(lambda_=1.0, theta=1.0, eta_max=2.0))
        t1 = make_task("t1", deadline=10.0)

        with self.assertRaises(KeyError):
            model.evaluate(tasks=[t1], completion_times={})

    def test_zero_deadline_rejected(self) -> None:
        model = ExponentialSLAPenaltyModel(SLAParameters(lambda_=1.0, theta=1.0, eta_max=2.0))
        t1 = make_task("t1", deadline=0.0)

        with self.assertRaises(ValueError):
            model.evaluate(tasks=[t1], completion_times={"t1": 1.0})


if __name__ == "__main__":
    unittest.main()
