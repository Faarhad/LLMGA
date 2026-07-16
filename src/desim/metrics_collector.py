from dataclasses import dataclass
from typing import Dict, List

from .energy import DatacenterEnergyBreakdown, QuadraticEnergyModel
from .event import Event
from .fairness import FairnessModel, FairnessParameters, FairnessResult
from .models import CloudConfiguration
from .simulation import Simulation
from .sla import ExponentialSLAPenaltyModel, SLAAggregateResult, SLAParameters
from .state import SimulationState
from .utilization import UtilizationSnapshot, UtilizationTracker


@dataclass(frozen=True)
class FitnessParameters:
    w_energy: float
    w_sla: float
    xi: float
    energy_norm_max: float = 1.0
    sla_norm_max: float = 1.0

    def __post_init__(self) -> None:
        if self.w_energy < 0 or self.w_sla < 0:
            raise ValueError("w_energy and w_sla must be >= 0")
        if abs((self.w_energy + self.w_sla) - 1.0) > 1e-9:
            raise ValueError("w_energy + w_sla must equal 1")
        if self.xi < 0:
            raise ValueError("xi must be >= 0")
        if self.energy_norm_max <= 0 or self.sla_norm_max <= 0:
            raise ValueError("normalization maxima must be > 0")


@dataclass(frozen=True)
class MetricsSnapshot:
    makespan: float
    throughput: float
    energy: DatacenterEnergyBreakdown
    average_waiting_time: float
    average_response_time: float
    sla: SLAAggregateResult
    fairness: FairnessResult
    utilization: UtilizationSnapshot
    average_utilization_per_vm: Dict[str, float]
    cpu_occupancy: float
    fitness: float


class MetricsCollector:
    """Event-observing collector that computes simulation metrics post-run."""

    def __init__(
        self,
        configuration: CloudConfiguration,
        utilization_tracker: UtilizationTracker,
        energy_model: QuadraticEnergyModel | None = None,
        sla_model: ExponentialSLAPenaltyModel | None = None,
        fairness_model: FairnessModel | None = None,
        fitness_parameters: FitnessParameters | None = None,
    ) -> None:
        self.configuration = configuration
        self.utilization_tracker = utilization_tracker
        self.energy_model = energy_model or QuadraticEnergyModel()
        self.sla_model = sla_model or ExponentialSLAPenaltyModel(
            SLAParameters(lambda_=1.0, theta=1.0, eta_max=2.0)
        )
        self.fairness_model = fairness_model or FairnessModel(
            FairnessParameters(omega_1=0.5, omega_2=0.5, mu=1.0)
        )
        self.fitness_parameters = fitness_parameters or FitnessParameters(
            w_energy=0.5,
            w_sla=0.5,
            xi=1.0,
            energy_norm_max=1.0,
            sla_norm_max=1.0,
        )

        self._completion_times: Dict[str, float] = {}
        self._waiting_times: List[float] = []
        self._response_times: List[float] = []
        self._makespan: float = 0.0
        self._finished_tasks: int = 0

    def register(self, simulation: Simulation, state_key: str = "metrics_collector") -> None:
        simulation.state.set(state_key, self)
        simulation.dispatcher.register("vm.task_finished", self._on_task_finished)

    def _on_task_finished(self, state: SimulationState, event: Event) -> None:
        task = event.payload["task"]
        finish_time = float(event.payload["finish_time"])
        start_time = float(event.payload["start_time"])
        arrival_time = float(event.payload["arrival_time"])

        waiting_time = start_time - arrival_time
        response_time = finish_time - arrival_time

        self._completion_times[task.task_id] = finish_time
        self._waiting_times.append(waiting_time)
        self._response_times.append(response_time)
        self._finished_tasks += 1
        if finish_time > self._makespan:
            self._makespan = finish_time

    def finalize(self) -> MetricsSnapshot:
        epoch_length = self.configuration.epoch_length
        utilization = self.utilization_tracker.snapshot()

        energy = self.energy_model.compute_datacenter_energy(
            datacenter=self.configuration.datacenter,
            utilization_snapshot=utilization,
            epoch_length=epoch_length,
        )

        completion_times = self._completion_times_with_fallback(default_time=epoch_length)
        sla = self.sla_model.evaluate(
            tasks=self.configuration.tasks,
            completion_times=completion_times,
        )
        fairness = self.fairness_model.evaluate(sla)

        average_waiting_time = sum(self._waiting_times) / len(self._waiting_times) if self._waiting_times else 0.0
        average_response_time = sum(self._response_times) / len(self._response_times) if self._response_times else 0.0
        throughput = (self._finished_tasks / self._makespan) if self._makespan > 0 else 0.0

        avg_util_per_vm, cpu_occupancy = self._utilization_metrics(utilization, epoch_length)

        sla_objective = sla.aggregate_penalty + (self.fitness_parameters.xi * fairness.combined_fairness)
        fitness = (
            self.fitness_parameters.w_energy * (energy.total_energy / self.fitness_parameters.energy_norm_max)
            + self.fitness_parameters.w_sla * (sla_objective / self.fitness_parameters.sla_norm_max)
        )

        return MetricsSnapshot(
            makespan=self._makespan,
            throughput=throughput,
            energy=energy,
            average_waiting_time=average_waiting_time,
            average_response_time=average_response_time,
            sla=sla,
            fairness=fairness,
            utilization=utilization,
            average_utilization_per_vm=avg_util_per_vm,
            cpu_occupancy=cpu_occupancy,
            fitness=fitness,
        )

    def _completion_times_with_fallback(self, default_time: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for task in self.configuration.tasks:
            out[task.task_id] = self._completion_times.get(task.task_id, default_time)
        return out

    def _utilization_metrics(self, utilization: UtilizationSnapshot, epoch_length: float) -> tuple[Dict[str, float], float]:
        if epoch_length <= 0:
            raise ValueError("epoch_length must be > 0")

        vm_cpu = {vm.vm_id: vm.capacity.cpu_mips for vm in self.configuration.datacenter.virtual_machines}
        avg_util_per_vm: Dict[str, float] = {}

        total_used_cpu_time = 0.0
        total_cpu_capacity_time = 0.0

        for vm_id, trace in utilization.traces.items():
            area = 0.0
            for interval in trace.intervals:
                area += interval.utilization * (interval.end_time - interval.start_time)

            avg_util_per_vm[vm_id] = area / epoch_length

            cpu = vm_cpu.get(vm_id, 0.0)
            total_used_cpu_time += area * cpu
            total_cpu_capacity_time += epoch_length * cpu

        cpu_occupancy = (total_used_cpu_time / total_cpu_capacity_time) if total_cpu_capacity_time > 0 else 0.0
        return avg_util_per_vm, cpu_occupancy
