import importlib

from desim.algorithms.base import Scheduler
from desim.framework.configuration import AppConfigLoader
from desim.framework.dataset_loading import DatasetLoader
from desim.framework.orchestrator import SimulationOrchestrator


def build_scheduler(config):
    name = config.scheduler.name.strip()
    if not name:
        raise ValueError("scheduler.name must be a non-empty algorithm module name")

    options = config.scheduler.options
    seed = options.get("seed", config.random_seeds.scheduler_seed)
    module_name = name[:-3] if name.endswith(".py") else name
    module = importlib.import_module(f"desim.algorithms.{module_name}")
    factory = getattr(module, "create_scheduler", None)

    if not callable(factory):
        raise ValueError(
            f"algorithm module '{name}' must define create_scheduler(options, seed)"
        )

    scheduler = factory(options=options, seed=seed)
    if not isinstance(scheduler, Scheduler):
        raise ValueError(
            f"algorithm module '{name}' create_scheduler(...) must return a Scheduler instance"
        )
    return scheduler


def main() -> None:
    config = AppConfigLoader().load_from_yaml("examples/ga/ga_config.yaml")
    dataset = DatasetLoader().load(config.dataset.source)
    seed_count = 50
    totals = {
        "makespan": 0.0,
        "throughput": 0.0,
        "energy": 0.0,
        "waiting_time": 0.0,
        "response_time": 0.0,
        "sla_penalty": 0.0,
        "jain_fairness": 0.0,
        "unfairness": 0.0,
        "utilization": 0.0,
        "fitness": 0.0,
    }

    for seed in range(1, seed_count + 1):
        config.scheduler.options["seed"] = seed
        scheduler = build_scheduler(config)
        orchestrator = SimulationOrchestrator(scheduler=scheduler)
        state = orchestrator.run(dataset)
        metrics = state.get("metrics")

        totals["makespan"] += metrics.makespan
        totals["throughput"] += metrics.throughput
        totals["energy"] += metrics.energy.total_energy
        totals["waiting_time"] += metrics.average_waiting_time
        totals["response_time"] += metrics.average_response_time
        totals["sla_penalty"] += metrics.sla.aggregate_penalty
        totals["jain_fairness"] += metrics.fairness.jain_index
        totals["unfairness"] += metrics.fairness.combined_fairness
        totals["utilization"] += metrics.cpu_occupancy
        totals["fitness"] += metrics.fitness

        print(f"Seed: {seed}")
        print("Makespan:", metrics.makespan)
        print("SLA penalty:", metrics.sla.aggregate_penalty)
        print("Jain Fairness (higher better):", metrics.fairness.jain_index)
        print("Unfairness (lower better):", metrics.fairness.combined_fairness)
        print("Fitness:", metrics.fitness)
        print("-" * 50)

    print("Average over seeds 1..50")
    print("Avg Makespan:", totals["makespan"] / seed_count)
    print("Avg Throughput:", totals["throughput"] / seed_count)
    print("Avg Energy:", totals["energy"] / seed_count)
    print("Avg Waiting time:", totals["waiting_time"] / seed_count)
    print("Avg Response time:", totals["response_time"] / seed_count)
    print("Avg SLA penalty:", totals["sla_penalty"] / seed_count)
    print("Avg Jain Fairness (higher better):", totals["jain_fairness"] / seed_count)
    print("Avg Unfairness (lower better):", totals["unfairness"] / seed_count)
    print("Avg Utilization:", totals["utilization"] / seed_count)
    print("Avg Fitness:", totals["fitness"] / seed_count)


if __name__ == "__main__":
    main()
