import unittest

from desim.framework.models import (
    CloudConfiguration,
    Datacenter,
    PhysicalMachine,
    PowerProfile,
    ResourceCapacity,
    Task,
    VirtualMachine,
)


class TestResourceCapacity(unittest.TestCase):
    def test_create(self) -> None:
        c = ResourceCapacity(cpu_mips=1000, memory_mb=2048, bandwidth_mbps=100)
        self.assertEqual(c.cpu_mips, 1000)

    def test_negative_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResourceCapacity(cpu_mips=-1, memory_mb=1, bandwidth_mbps=1)


class TestPowerProfile(unittest.TestCase):
    def test_create(self) -> None:
        p = PowerProfile(idle_watts=40, max_watts=100)
        self.assertEqual(p.max_watts, 100)

    def test_invalid_order_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PowerProfile(idle_watts=100, max_watts=40)


class TestPhysicalMachine(unittest.TestCase):
    def test_create(self) -> None:
        pm = PhysicalMachine(
            machine_id="pm1",
            capacity=ResourceCapacity(2000, 8192, 1000),
            base_power_watts=120,
        )
        self.assertEqual(pm.machine_id, "pm1")

    def test_invalid_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PhysicalMachine(
                machine_id="",
                capacity=ResourceCapacity(1, 1, 1),
                base_power_watts=10,
            )


class TestVirtualMachine(unittest.TestCase):
    def test_create(self) -> None:
        vm = VirtualMachine(
            vm_id="vm1",
            vm_type="small",
            host_machine_id="pm1",
            capacity=ResourceCapacity(1000, 2048, 200),
            power=PowerProfile(20, 60),
            vcpu_count=2,
            availability_time=0,
        )
        self.assertEqual(vm.vm_id, "vm1")

    def test_invalid_vcpu_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VirtualMachine(
                vm_id="vm1",
                vm_type="small",
                host_machine_id="pm1",
                capacity=ResourceCapacity(1000, 2048, 200),
                power=PowerProfile(20, 60),
                vcpu_count=0,
            )


class TestTask(unittest.TestCase):
    def test_create(self) -> None:
        task = Task(
            task_id="t1",
            workload_mi=5000,
            arrival_time=0,
            deadline=10,
            cpu_demand_mips=200,
            memory_demand_mb=512,
            io_size_mb=50,
        )
        self.assertEqual(task.task_id, "t1")

    def test_invalid_workload_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Task(
                task_id="t1",
                workload_mi=-1,
                arrival_time=0,
                deadline=10,
                cpu_demand_mips=200,
                memory_demand_mb=512,
                io_size_mb=50,
            )


class TestDatacenter(unittest.TestCase):
    def test_create(self) -> None:
        pm = PhysicalMachine("pm1", ResourceCapacity(2000, 8192, 1000), 120)
        vm = VirtualMachine(
            vm_id="vm1",
            vm_type="small",
            host_machine_id="pm1",
            capacity=ResourceCapacity(1000, 2048, 200),
            power=PowerProfile(20, 60),
            vcpu_count=2,
        )
        dc = Datacenter("dc1", [pm], [vm])
        self.assertEqual(dc.datacenter_id, "dc1")

    def test_unknown_vm_host_rejected(self) -> None:
        pm = PhysicalMachine("pm1", ResourceCapacity(2000, 8192, 1000), 120)
        vm = VirtualMachine(
            vm_id="vm1",
            vm_type="small",
            host_machine_id="pm2",
            capacity=ResourceCapacity(1000, 2048, 200),
            power=PowerProfile(20, 60),
            vcpu_count=2,
        )
        with self.assertRaises(ValueError):
            Datacenter("dc1", [pm], [vm])


class TestCloudConfiguration(unittest.TestCase):
    def test_create(self) -> None:
        pm = PhysicalMachine("pm1", ResourceCapacity(2000, 8192, 1000), 120)
        vm = VirtualMachine(
            vm_id="vm1",
            vm_type="small",
            host_machine_id="pm1",
            capacity=ResourceCapacity(1000, 2048, 200),
            power=PowerProfile(20, 60),
            vcpu_count=2,
        )
        dc = Datacenter("dc1", [pm], [vm])
        task = Task("t1", 1000, 0, 10, 100, 128, 10)

        cfg = CloudConfiguration(datacenter=dc, tasks=[task], epoch_length=20)
        self.assertEqual(cfg.epoch_length, 20)

    def test_non_positive_epoch_rejected(self) -> None:
        pm = PhysicalMachine("pm1", ResourceCapacity(2000, 8192, 1000), 120)
        vm = VirtualMachine(
            vm_id="vm1",
            vm_type="small",
            host_machine_id="pm1",
            capacity=ResourceCapacity(1000, 2048, 200),
            power=PowerProfile(20, 60),
            vcpu_count=2,
        )
        dc = Datacenter("dc1", [pm], [vm])

        with self.assertRaises(ValueError):
            CloudConfiguration(datacenter=dc, tasks=[], epoch_length=0)


if __name__ == "__main__":
    unittest.main()

