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
        pm = make_pm(base_power_watts=100.0)
        vm = make_vm("vm1", host_machine_id=pm.machine_id, cpu_mips=1000.0)
        task = make_task("t1", workload_mi=1000.0, arrival_time=0.0, deadline=5.0, cpu_demand_mips=500.0)
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
        expected_total_energy = expected_vm_energy + (100.0 * 10.0)

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

    def test_two_vms_with_no_queueing(self) -> None:
        pm = make_pm()
        vm1 = make_vm("vm1", host_machine_id=pm.machine_id)
        vm2 = make_vm("vm2", host_machine_id=pm.machine_id)
        t1 = make_task("t1", 1000.0, 0.0, 5.0, 500.0)
        t2 = make_task("t2", 1000.0, 0.0, 5.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm1, vm2], [t1, t2])

        metrics = run_scenario(dataset, {"t1": "vm1", "t2": "vm2"})

        self.assert_metric("avg_waiting_time", metrics.average_waiting_time, 0.0)
        self.assert_metric("avg_response_time", metrics.average_response_time, 1.0)
        self.assert_metric("makespan", metrics.makespan, 1.0)
        self.assert_metric("throughput", metrics.throughput, 2.0 / 1.0)

    def test_vm_with_carry_over_availability(self) -> None:
        pm = make_pm()
        vm = make_vm("vm1", host_machine_id=pm.machine_id, availability_time=3.0)
        task = make_task("t1", 1000.0, 0.0, 10.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm], [task])

        metrics = run_scenario(dataset, {"t1": "vm1"})
        task_metrics = metrics.sla.per_task["t1"]
        timing = metrics.utilization.traces["vm1"].intervals

        expected_waiting = 3.0
        expected_response = 4.0
        expected_carry_over = 3.0

        self.assert_metric("waiting_time", metrics.average_waiting_time, expected_waiting)
        self.assert_metric("response_time", metrics.average_response_time, expected_response)
        self.assert_metric("completion_time", task_metrics.completion_time, 4.0)
        self.assert_metric("carry_over_idle_end", timing[0].end_time, expected_carry_over)

    def test_pm_base_energy_validation(self) -> None:
        pm = make_pm(base_power_watts=120.0)
        vm = make_vm("vm1", host_machine_id=pm.machine_id)
        task = make_task("t1", 1000.0, 0.0, 5.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm], [task])

        metrics = run_scenario(dataset, {"t1": "vm1"})
        expected_pm_base_energy = 120.0 * 10.0

        self.assert_metric("pm_base_energy", metrics.energy.total_pm_base_energy, expected_pm_base_energy)

    def test_vm_dynamic_energy_validation(self) -> None:
        pm = make_pm()
        vm = make_vm("vm1", host_machine_id=pm.machine_id, idle_watts=20.0, max_watts=60.0)
        task = make_task("t1", 2000.0, 0.0, 5.0, 250.0)
        dataset = build_configuration(10.0, [pm], [vm], [task])

        metrics = run_scenario(dataset, {"t1": "vm1"})

        expected_duration = 2.0
        expected_utilization = 0.25
        expected_dynamic = linear_dynamic_energy(20.0, 60.0, expected_utilization, expected_duration)

        self.assert_metric("vm_dynamic_energy", metrics.energy.vm_energy["vm1"].dynamic_energy, expected_dynamic)

    def test_waiting_time_validation(self) -> None:
        pm = make_pm()
        vm = make_vm("vm1", host_machine_id=pm.machine_id)
        t1 = make_task("t1", 1000.0, 0.0, 5.0, 500.0)
        t2 = make_task("t2", 1000.0, 0.5, 5.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm], [t1, t2])

        metrics = run_scenario(dataset, {"t1": "vm1", "t2": "vm1"})
        expected_waiting_avg = (0.0 + 0.5) / 2.0

        self.assert_metric("avg_waiting_time", metrics.average_waiting_time, expected_waiting_avg)

    def test_response_time_validation(self) -> None:
        pm = make_pm()
        vm = make_vm("vm1", host_machine_id=pm.machine_id)
        t1 = make_task("t1", 1000.0, 0.0, 5.0, 500.0)
        t2 = make_task("t2", 1000.0, 0.5, 5.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm], [t1, t2])

        metrics = run_scenario(dataset, {"t1": "vm1", "t2": "vm1"})
        expected_response_avg = ((1.0 - 0.0) + (2.0 - 0.5)) / 2.0

        self.assert_metric("avg_response_time", metrics.average_response_time, expected_response_avg)

    def test_makespan_validation(self) -> None:
        pm = make_pm()
        vm1 = make_vm("vm1", host_machine_id=pm.machine_id)
        vm2 = make_vm("vm2", host_machine_id=pm.machine_id)
        t1 = make_task("t1", 3000.0, 0.0, 10.0, 500.0)
        t2 = make_task("t2", 1000.0, 0.0, 10.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm1, vm2], [t1, t2])

        metrics = run_scenario(dataset, {"t1": "vm1", "t2": "vm2"})
        expected_makespan = 3.0

        self.assert_metric("makespan", metrics.makespan, expected_makespan)

    def test_sla_penalty_validation(self) -> None:
        pm = make_pm()
        vm = make_vm("vm1", host_machine_id=pm.machine_id)
        task = make_task("t1", 3000.0, 0.0, 2.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm], [task])

        metrics = run_scenario(dataset, {"t1": "vm1"})
        expected_completion = 3.0
        expected_penalty = exponential_sla_penalty(expected_completion, 2.0)

        self.assert_metric("deadline_violation", metrics.sla.per_task["t1"].deadline_violation, 1.0)
        self.assert_metric("normalized_violation", metrics.sla.per_task["t1"].normalized_violation, 0.5)
        self.assert_metric("sla_aggregate_penalty", metrics.sla.aggregate_penalty, expected_penalty)

    def test_jain_fairness_validation(self) -> None:
        pm = make_pm()
        vm1 = make_vm("vm1", host_machine_id=pm.machine_id)
        vm2 = make_vm("vm2", host_machine_id=pm.machine_id)
        t1 = make_task("t1", 3000.0, 0.0, 2.0, 500.0)
        t2 = make_task("t2", 4000.0, 0.0, 2.0, 500.0)
        dataset = build_configuration(10.0, [pm], [vm1, vm2], [t1, t2])

        metrics = run_scenario(dataset, {"t1": "vm1", "t2": "vm2"})

        penalty_1 = exponential_sla_penalty(3.0, 2.0)
        penalty_2 = exponential_sla_penalty(4.0, 2.0)
        expected_jain = jain_index([penalty_1, penalty_2])

        self.assert_metric("jain_index", metrics.fairness.jain_index, expected_jain)


if __name__ == "__main__":
    unittest.main()
