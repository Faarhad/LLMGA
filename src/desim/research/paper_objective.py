from typing import Protocol


class FitnessParametersLike(Protocol):
    w_energy: float
    w_sla: float
    xi: float
    energy_norm_max: float
    sla_norm_max: float


def compute_sla_objective(aggregate_sla_penalty: float, combined_fairness: float, xi: float) -> float:
    return aggregate_sla_penalty + (xi * combined_fairness)


def compute_paper_fitness(
    energy_total: float,
    aggregate_sla_penalty: float,
    combined_fairness: float,
    fitness_parameters: FitnessParametersLike,
) -> float:
    sla_objective = compute_sla_objective(
        aggregate_sla_penalty=aggregate_sla_penalty,
        combined_fairness=combined_fairness,
        xi=fitness_parameters.xi,
    )
    return (
        fitness_parameters.w_energy * (energy_total / fitness_parameters.energy_norm_max)
        + fitness_parameters.w_sla * (sla_objective / fitness_parameters.sla_norm_max)
    )
