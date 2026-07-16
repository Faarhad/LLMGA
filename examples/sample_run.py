from desim.framework.configuration import AppConfigLoader
from desim.framework.dataset_loading import DatasetLoader
from desim.framework.orchestrator import SimulationOrchestrator
from desim.algorithms.scheduling import RandomScheduler


def main() -> None:
    config = AppConfigLoader().load_from_yaml("examples/sample_config.yaml")
    dataset = DatasetLoader().load(config.dataset.source)

    scheduler_seed = config.random_seeds.scheduler_seed
    scheduler = RandomScheduler(seed=scheduler_seed)
    orchestrator = SimulationOrchestrator(scheduler=scheduler)

    state = orchestrator.run(dataset)
    metrics = state.get("metrics")

    print("Makespan:", metrics.makespan)
    print("Throughput:", metrics.throughput)
    print("Energy:", metrics.energy.total_energy)
    print("Waiting time:", metrics.average_waiting_time)
    print("Response time:", metrics.average_response_time)
    print("SLA penalty:", metrics.sla.aggregate_penalty)
    print("Fairness:", metrics.fairness.combined_fairness)
    print("Utilization:", metrics.cpu_occupancy)
    print("Fitness:", metrics.fitness)


if __name__ == "__main__":
    main()

