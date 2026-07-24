from typing import Dict

from desim.framework.orchestrator import SimulationOrchestrator
from desim.algorithms.scheduling import Scheduler, SchedulingResult

class FixedScheduler(Scheduler):
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = dict(mapping)

  def schedule(self, waiting_tasks, vm_states):
    waiting_ids = {task.task_id for task in waiting_tasks}
    filtered = {task_id: vm_id for task_id, vm_id in self.mapping.items() if task_id in waiting_ids}
    return SchedulingResult(task_to_vm=filtered)


def main() -> None:
    dataset = {
  "epoch_length": 10.0,
  "datacenter": {
    "datacenter_id": "dc_test5",
    "physical_machines": [
      {
        "machine_id": "pm1",
        "cpu_mips": 10000,
        "memory_mb": 65536,
        "bandwidth_mbps": 10000,
        "base_power_watts": 100.0
      },
      {
        "machine_id": "pm2",
        "cpu_mips": 12000,
        "memory_mb": 65536,
        "bandwidth_mbps": 10000,
        "base_power_watts": 120.0
      }
    ],
    "virtual_machines": [
      {
        "vm_id": "vm1",
        "vm_type": "small",
        "host_machine_id": "pm1",
        "cpu_mips": 1000.0,
        "memory_mb": 2048,
        "bandwidth_mbps": 100,
        "idle_watts": 20.0,
        "max_watts": 60.0,
        "vcpu_count": 2,
        "availability_time": 0.0
      },
      {
        "vm_id": "vm2",
        "vm_type": "medium",
        "host_machine_id": "pm2",
        "cpu_mips": 2000.0,
        "memory_mb": 4096,
        "bandwidth_mbps": 200,
        "idle_watts": 15.0,
        "max_watts": 75.0,
        "vcpu_count": 2,
        "availability_time": 0.0
      }
    ]
  },
  "tasks": [
    {
      "task_id": "t1",
      "workload_mi": 2000.0,
      "arrival_time": 0.0,
      "deadline": 10.0,
      "cpu_demand_mips": 500.0,
      "memory_demand_mb": 128,
      "io_size_mb": 10
    },
    {
      "task_id": "t2",
      "workload_mi": 3000.0,
      "arrival_time": 0.0,
      "deadline": 10.0,
      "cpu_demand_mips": 1200.0,
      "memory_demand_mb": 128,
      "io_size_mb": 10
    }
  ]
}

    fixed_mapping = {
    "t1": "vm1",
    "t2": "vm2"
}
    
    orchestrator = SimulationOrchestrator(scheduler=FixedScheduler(fixed_mapping))
    state = orchestrator.run(dataset)
    metrics = state.get("metrics")

    print("=== 2 Task / 2 VM Example ===")
    print("Task mapping: t1 -> vm1, t2 -> vm2")
    print("\n--- Simulator Output ---")
    print(f"waiting_avg: {metrics.average_waiting_time}")
    print(f"response_avg: {metrics.average_response_time}")
    print(f"makespan: {metrics.makespan}")
    print(f"throughput: {metrics.throughput}")
    print(f"vm_total_energy: {metrics.energy.total_vm_energy}")
    print(f"pm_base_energy: {metrics.energy.total_pm_base_energy}")
    print(f"total_energy: {metrics.energy.total_energy}")
    print(f"sla_penalty: {metrics.sla.aggregate_penalty}")


if __name__ == "__main__":
    main()

