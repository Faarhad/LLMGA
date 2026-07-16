from dataclasses import dataclass
from typing import Dict, Protocol

from .models import Datacenter, PhysicalMachine, VirtualMachine
from .utilization import UtilizationInterval, UtilizationSnapshot


@dataclass(frozen=True)
class VmPowerCoefficients:
    """Quadratic power coefficients in P = Pidle + alpha*u + beta*u^2."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be >= 0")


class CoefficientProvider(Protocol):
    def for_vm(self, vm: VirtualMachine) -> VmPowerCoefficients:
        ...


@dataclass(frozen=True)
class FixedCoefficientProvider:
    """VM-specific coefficient provider."""

    coefficients: Dict[str, VmPowerCoefficients]

    def for_vm(self, vm: VirtualMachine) -> VmPowerCoefficients:
        if vm.vm_id not in self.coefficients:
            raise KeyError(f"missing coefficients for vm_id={vm.vm_id}")
        coeff = self.coefficients[vm.vm_id]
        delta = vm.power.max_watts - vm.power.idle_watts
        if abs((coeff.alpha + coeff.beta) - delta) > 1e-9:
            raise ValueError(
                f"invalid coefficients for vm_id={vm.vm_id}: alpha+beta must equal Pmax-Pidle"
            )
        return coeff


@dataclass(frozen=True)
class CornerPointCalibrationProvider:
    """Automatic calibration enforcing alpha + beta = Pmax - Pidle."""

    alpha_share: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha_share < 0 or self.alpha_share > 1:
            raise ValueError("alpha_share must be in [0, 1]")

    def for_vm(self, vm: VirtualMachine) -> VmPowerCoefficients:
        delta = vm.power.max_watts - vm.power.idle_watts
        alpha = self.alpha_share * delta
        beta = delta - alpha
        return VmPowerCoefficients(alpha=alpha, beta=beta)


@dataclass(frozen=True)
class VmEnergyBreakdown:
    vm_id: str
    epoch_length: float
    idle_time: float
    active_time: float
    idle_energy: float
    active_static_energy: float
    dynamic_energy: float
    vm_energy: float


@dataclass(frozen=True)
class PmBaseEnergyBreakdown:
    machine_id: str
    epoch_length: float
    base_power_watts: float
    base_energy: float


@dataclass(frozen=True)
class DatacenterEnergyBreakdown:
    vm_energy: Dict[str, VmEnergyBreakdown]
    pm_base_energy: Dict[str, PmBaseEnergyBreakdown]
    total_vm_energy: float
    total_pm_base_energy: float
    total_energy: float


class QuadraticEnergyModel:
    """Computes energy by integrating utilization intervals over the epoch."""

    def __init__(self, coefficient_provider: CoefficientProvider | None = None) -> None:
        self._coeff_provider = coefficient_provider or CornerPointCalibrationProvider(alpha_share=1.0)

    def compute_vm_energy(
        self,
        vm: VirtualMachine,
        intervals: list[UtilizationInterval],
        epoch_length: float,
    ) -> VmEnergyBreakdown:
        if epoch_length <= 0:
            raise ValueError("epoch_length must be > 0")

        coeff = self._coeff_provider.for_vm(vm)

        idle_time = 0.0
        active_time = 0.0
        dynamic_energy = 0.0

        for interval in intervals:
            if interval.vm_id != vm.vm_id:
                raise ValueError("interval vm_id does not match vm")
            if interval.start_time < 0 or interval.end_time > epoch_length:
                raise ValueError("interval must be within [0, epoch_length]")
            if interval.utilization < 0 or interval.utilization > 1:
                raise ValueError("utilization must be in [0, 1]")

            duration = interval.end_time - interval.start_time
            if duration < 0:
                raise ValueError("interval duration must be >= 0")

            if interval.utilization == 0:
                idle_time += duration
            else:
                active_time += duration

            dynamic_energy += (
                coeff.alpha * interval.utilization + coeff.beta * interval.utilization * interval.utilization
            ) * duration

        total_observed = idle_time + active_time
        if total_observed > epoch_length + 1e-9:
            raise ValueError("interval durations exceed epoch length")

        idle_energy = vm.power.idle_watts * idle_time
        active_static_energy = vm.power.idle_watts * active_time
        vm_energy = idle_energy + active_static_energy + dynamic_energy

        return VmEnergyBreakdown(
            vm_id=vm.vm_id,
            epoch_length=epoch_length,
            idle_time=idle_time,
            active_time=active_time,
            idle_energy=idle_energy,
            active_static_energy=active_static_energy,
            dynamic_energy=dynamic_energy,
            vm_energy=vm_energy,
        )

    @staticmethod
    def compute_pm_base_energy(pm: PhysicalMachine, epoch_length: float) -> PmBaseEnergyBreakdown:
        if epoch_length <= 0:
            raise ValueError("epoch_length must be > 0")
        base_energy = pm.base_power_watts * epoch_length
        return PmBaseEnergyBreakdown(
            machine_id=pm.machine_id,
            epoch_length=epoch_length,
            base_power_watts=pm.base_power_watts,
            base_energy=base_energy,
        )

    def compute_datacenter_energy(
        self,
        datacenter: Datacenter,
        utilization_snapshot: UtilizationSnapshot,
        epoch_length: float,
    ) -> DatacenterEnergyBreakdown:
        vm_breakdown: Dict[str, VmEnergyBreakdown] = {}
        pm_breakdown: Dict[str, PmBaseEnergyBreakdown] = {}

        total_vm_energy = 0.0
        for vm in datacenter.virtual_machines:
            if vm.vm_id not in utilization_snapshot.traces:
                raise KeyError(f"missing utilization trace for vm_id={vm.vm_id}")
            intervals = utilization_snapshot.traces[vm.vm_id].intervals
            vm_energy = self.compute_vm_energy(vm=vm, intervals=intervals, epoch_length=epoch_length)
            vm_breakdown[vm.vm_id] = vm_energy
            total_vm_energy += vm_energy.vm_energy

        total_pm_base_energy = 0.0
        for pm in datacenter.physical_machines:
            pm_energy = self.compute_pm_base_energy(pm=pm, epoch_length=epoch_length)
            pm_breakdown[pm.machine_id] = pm_energy
            total_pm_base_energy += pm_energy.base_energy

        total_energy = total_vm_energy + total_pm_base_energy
        return DatacenterEnergyBreakdown(
            vm_energy=vm_breakdown,
            pm_base_energy=pm_breakdown,
            total_vm_energy=total_vm_energy,
            total_pm_base_energy=total_pm_base_energy,
            total_energy=total_energy,
        )
