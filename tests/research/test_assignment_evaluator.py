import unittest

from desim.algorithms.base import VmQueuedTaskView, VmRunningTaskView, VmStateView
from desim.framework.models import Datacenter, PhysicalMachine, PowerProfile, ResourceCapacity, Task, VirtualMachine
from desim.research.assignment_evaluator import AssignmentEvaluator
from desim.research.energy import CornerPointCalibrationProvider, QuadraticEnergyModel
from desim.research.fairness import FairnessModel, FairnessParameters
from desim.research.metrics_collector import FitnessParameters
from desim.research.sla import ExponentialSLAPenaltyModel, SLAParameters


def _pm() -> PhysicalMachine:
    return PhysicalMachine(
        machine_id="pm1",
        capacity=ResourceCapacity(cpu_mips=5000, memory_mb=32768, bandwidth_mbps=10000),
        base_power_watts=100.0,
    )


def _vm(vm_id: str) -> VirtualMachine:
    return VirtualMachine(
        vm_id=vm_id,
        vm_type="small",
        host_machine_id="pm1",
        capacity=ResourceCapacity(cpu_mips=1000, memory_mb=2048, bandwidth_mbps=100),
        power=PowerProfile(idle_watts=20.0, max_watts=60.0),
        vcpu_count=2,
        availability_time=0.0,
    )


def _task(task_id: str, workload_mi: float, arrival: float, deadline: float, cpu_demand: float) -> Task:
    return Task(
        task_id=task_id,
        workload_mi=workload_mi,
        arrival_time=arrival,
        deadline=deadline,
        cpu_demand_mips=cpu_demand,
        memory_demand_mb=128.0,
        io_size_mb=10.0,
    )


class TestAssignmentEvaluator(unittest.TestCase):
    def test_respects_running_and_fifo_queued_tasks(self) -> None:
        vm1 = _vm("vm1")
        vm2 = _vm("vm2")
        dc = Datacenter(datacenter_id="dc1", physical_machines=[_pm()], virtual_machines=[vm1, vm2])

        running = _task("run", workload_mi=1000.0, arrival=0.0, deadline=20.0, cpu_demand=500.0)
        queued = _task("q1", workload_mi=1000.0, arrival=0.0, deadline=20.0, cpu_demand=500.0)
        new_task = _task("new", workload_mi=1000.0, arrival=0.0, deadline=20.0, cpu_demand=500.0)

        vm_states = [
            VmStateView(
                vm=vm1,
                availability_time=2.0,
                queue=[VmQueuedTaskView(task=queued, remaining_duration=1.0)],
                running_task=VmRunningTaskView(task=running, remaining_duration=1.0),
            ),
            VmStateView(vm=vm2, availability_time=1.0),
        ]

        evaluator = AssignmentEvaluator(
            datacenter=dc,
            energy_model=QuadraticEnergyModel(
                coefficient_provider=CornerPointCalibrationProvider(alpha_share=1.0)
            ),
            sla_model=ExponentialSLAPenaltyModel(SLAParameters(lambda_=1.0, theta=1.0, eta_max=2.0)),
            fairness_model=FairnessModel(FairnessParameters(omega_1=0.5, omega_2=0.5, mu=1.0)),
            fitness_parameters=FitnessParameters(
                w_energy=0.5,
                w_sla=0.5,
                xi=1.0,
                energy_norm_max=1.0,
                sla_norm_max=1.0,
            ),
        )

        result = evaluator.evaluate(
            waiting_tasks=[new_task],
            vm_states=vm_states,
            assignment={"new": "vm1"},
            now=1.0,
        )

        vm1_order = [x.task.task_id for x in result.execution_plan["vm1"]]
        self.assertEqual(vm1_order, ["run", "q1", "new"])
        self.assertAlmostEqual(result.completion_times["run"], 2.0)
        self.assertAlmostEqual(result.completion_times["q1"], 3.0)
        self.assertAlmostEqual(result.completion_times["new"], 4.0)
