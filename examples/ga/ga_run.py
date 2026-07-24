import importlib

from desim.framework.configuration import AppConfigLoader
from desim.framework.dataset_loading import DatasetLoader
from desim.framework.orchestrator import SimulationOrchestrator
from desim.algorithms.base import Scheduler


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

    scheduler = build_scheduler(config)
    orchestrator = SimulationOrchestrator(scheduler=scheduler)

    state = orchestrator.run(dataset, app_config=config)
    metrics = state.get("metrics")

    print("Makespan:", metrics.makespan)
    print("Throughput:", metrics.throughput)
    print("Energy:", metrics.energy.total_energy)
    print("Waiting time:", metrics.average_waiting_time)
    print("Response time:", metrics.average_response_time)
    print("SLA penalty:", metrics.sla.aggregate_penalty)
    print("Jain Fairness (higher better):", metrics.fairness.jain_index)
    print("Unfairness (lower better):", metrics.fairness.combined_fairness)
    print("Utilization:", metrics.cpu_occupancy)
    print("Fitness:", metrics.fitness)


if __name__ == "__main__":
    main()

