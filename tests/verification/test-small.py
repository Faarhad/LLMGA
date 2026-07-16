import math
import unittest

from tests.verification.verification_utils import (
    TOLERANCE,
    build_configuration,
    exponential_sla_penalty,
    jain_index,
    linear_dynamic_energy,
    make_pm,
    make_task,
    make_vm,
    run_scenario,
)


class TestDeterministicVerification(unittest.TestCase):
    def assert_metric(self, metric_name: str, actual: float, expected: float) -> None:
        diff = abs(actual - expected)
        print(
            f"[{self._testMethodName}] {metric_name}: "
            f"expected={expected:.12g}, actual={actual:.12g}, diff={diff:.12g}"
        )
        self.assertAlmostEqual(actual, expected, delta=TOLERANCE)

    def test_single_task_on_one_vm(self) -> None:
        pm = make_pm(base_power_watts=70.0)
        vm = make_vm("vm1", host_machine_id=pm.machine_id, cpu_mips=1000.0)
        task = make_task("t1", workload_mi=1500.0, arrival_time=0.0, deadline=2.0, cpu_demand_mips=600.0)
        dataset = build_configuration(10.0, [pm], [vm], [task])

        metrics = run_scenario(dataset, {"t1": "vm1"})

        expected_execution = 1.0
        expected_waiting = 0.0
        expected_response = 1.0
        expected_makespan = 1.0
        expected_idle_energy = 20.0 * 9.0
        expected_active_static = 20.0 * 1.0
        expected_dynamic = linear_dynamic_energy(20.0, 60.0, 0.5, 1.0)
        expected_vm_energy = expected_idle_energy + expected_active_static + expected_dynamic
        expected_total_energy = expected_vm_energy + (80.0 * 10.0)

        self.assert_metric("waiting_time", metrics.average_waiting_time, expected_waiting)
        self.assert_metric("response_time", metrics.average_response_time, expected_response)
        self.assert_metric("makespan", metrics.makespan, expected_makespan)
        self.assert_metric("vm_energy", metrics.energy.total_vm_energy, expected_vm_energy)
        self.assert_metric("total_energy", metrics.energy.total_energy, expected_total_energy)
        self.assert_metric("utilization(vm1,active)", metrics.utilization.traces["vm1"].intervals[0].utilization, 0.5)
        self.assert_metric("execution_time", metrics.average_response_time, expected_execution)

    def test_two_sequential_tasks_on_one_vm(self) -> None:
        pm = make_pm()
        vm = make_vm("vm1", host_machine_id=pm.machine_id)
        t1 = make_task("t1", 1000.0, 0.0, 5.0, 500.0)
        t2 = make_task("t2", 1000.0, 0.0, 5.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm], [t1, t2])

        metrics = run_scenario(dataset, {"t1": "vm1", "t2": "vm1"})

        expected_waiting_avg = (0.0 + 1.0) / 2.0
        expected_response_avg = (1.0 + 2.0) / 2.0
        expected_makespan = 2.0

        self.assert_metric("avg_waiting_time", metrics.average_waiting_time, expected_waiting_avg)
        self.assert_metric("avg_response_time", metrics.average_response_time, expected_response_avg)
        self.assert_metric("makespan", metrics.makespan, expected_makespan)


if __name__ == "__main__":
    unittest.main()
