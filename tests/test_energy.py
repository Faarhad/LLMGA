import unittest

from desim.energy import (
    FixedCoefficientProvider,
    QuadraticEnergyModel,
    VmPowerCoefficients,
)
from desim.models import Datacenter, PhysicalMachine, PowerProfile, ResourceCapacity, VirtualMachine
from desim.utilization import UtilizationInterval, UtilizationSnapshot, VmUtilizationTrace


def make_vm(vm_id: str, host: str, idle: float, max_p: float, cpu: float = 1000) -> VirtualMachine:
    return VirtualMachine(
        vm_id=vm_id,
        vm_type="small",
        host_machine_id=host,
        capacity=ResourceCapacity(cpu_mips=cpu, memory_mb=2048, bandwidth_mbps=100),
        power=PowerProfile(idle_watts=idle, max_watts=max_p),
        vcpu_count=2,
    )


def make_pm(pm_id: str, base_power: float) -> PhysicalMachine:
    return PhysicalMachine(
        machine_id=pm_id,
        capacity=ResourceCapacity(cpu_mips=10000, memory_mb=65536, bandwidth_mbps=10000),
        base_power_watts=base_power,
    )


class TestQuadraticEnergyModel(unittest.TestCase):
    def test_vm_energy_integration_from_intervals(self) -> None:
        vm = make_vm("vm1", "pm1", idle=40.0, max_p=100.0)
        provider = FixedCoefficientProvider({"vm1": VmPowerCoefficients(alpha=20.0, beta=40.0)})
        model = QuadraticEnergyModel(coefficient_provider=provider)

        intervals = [
            UtilizationInterval(vm_id="vm1", start_time=0.0, end_time=5.0, utilization=0.0),
            UtilizationInterval(vm_id="vm1", start_time=5.0, end_time=10.0, utilization=0.5),
            UtilizationInterval(vm_id="vm1", start_time=10.0, end_time=20.0, utilization=0.0),
        ]
        out = model.compute_vm_energy(vm=vm, intervals=intervals, epoch_length=20.0)

        self.assertEqual(out.idle_time, 15.0)
        self.assertEqual(out.active_time, 5.0)
        self.assertEqual(out.idle_energy, 600.0)
        self.assertEqual(out.active_static_energy, 200.0)
        self.assertEqual(out.dynamic_energy, 100.0)
        self.assertEqual(out.vm_energy, 900.0)

    def test_pm_base_energy(self) -> None:
        pm = make_pm("pm1", base_power=100.0)
        out = QuadraticEnergyModel.compute_pm_base_energy(pm=pm, epoch_length=20.0)
        self.assertEqual(out.base_energy, 2000.0)

    def test_datacenter_total_energy(self) -> None:
        pm1 = make_pm("pm1", base_power=100.0)
        vm1 = make_vm("vm1", "pm1", idle=40.0, max_p=100.0)
        vm2 = make_vm("vm2", "pm1", idle=20.0, max_p=60.0)
        dc = Datacenter(datacenter_id="dc1", physical_machines=[pm1], virtual_machines=[vm1, vm2])

        provider = FixedCoefficientProvider(
            {
                "vm1": VmPowerCoefficients(alpha=20.0, beta=40.0),
                "vm2": VmPowerCoefficients(alpha=10.0, beta=30.0),
            }
        )
        model = QuadraticEnergyModel(coefficient_provider=provider)

        # vm1 energy = 900 from previous test construction.
        vm1_intervals = [
            UtilizationInterval(vm_id="vm1", start_time=0.0, end_time=5.0, utilization=0.0),
            UtilizationInterval(vm_id="vm1", start_time=5.0, end_time=10.0, utilization=0.5),
            UtilizationInterval(vm_id="vm1", start_time=10.0, end_time=20.0, utilization=0.0),
        ]
        # vm2: u=0.5 for 10s, idle 10s.
        # idle energy = 20*10 = 200
        # active static = 20*10 = 200
        # dynamic = (10*0.5 + 30*0.25)*10 = (5 + 7.5)*10 = 125
        # vm2 total = 525
        vm2_intervals = [
            UtilizationInterval(vm_id="vm2", start_time=0.0, end_time=10.0, utilization=0.5),
            UtilizationInterval(vm_id="vm2", start_time=10.0, end_time=20.0, utilization=0.0),
        ]

        snapshot = UtilizationSnapshot(
            traces={
                "vm1": VmUtilizationTrace(vm_id="vm1", current_utilization=0.0, last_change_time=20.0, intervals=vm1_intervals),
                "vm2": VmUtilizationTrace(vm_id="vm2", current_utilization=0.0, last_change_time=20.0, intervals=vm2_intervals),
            }
        )

        out = model.compute_datacenter_energy(datacenter=dc, utilization_snapshot=snapshot, epoch_length=20.0)
        self.assertEqual(out.vm_energy["vm1"].vm_energy, 900.0)
        self.assertEqual(out.vm_energy["vm2"].vm_energy, 525.0)
        self.assertEqual(out.total_vm_energy, 1425.0)
        self.assertEqual(out.total_pm_base_energy, 2000.0)
        self.assertEqual(out.total_energy, 3425.0)

    def test_missing_vm_coefficients_rejected(self) -> None:
        vm = make_vm("vm1", "pm1", idle=40.0, max_p=100.0)
        model = QuadraticEnergyModel(coefficient_provider=FixedCoefficientProvider({}))
        with self.assertRaises(KeyError):
            model.compute_vm_energy(vm=vm, intervals=[], epoch_length=20.0)

    def test_invalid_utilization_rejected(self) -> None:
        vm = make_vm("vm1", "pm1", idle=40.0, max_p=100.0)
        model = QuadraticEnergyModel(
            coefficient_provider=FixedCoefficientProvider({"vm1": VmPowerCoefficients(alpha=20.0, beta=40.0)})
        )
        with self.assertRaises(ValueError):
            model.compute_vm_energy(
                vm=vm,
                intervals=[UtilizationInterval(vm_id="vm1", start_time=0.0, end_time=1.0, utilization=1.2)],
                epoch_length=20.0,
            )


if __name__ == "__main__":
    unittest.main()
