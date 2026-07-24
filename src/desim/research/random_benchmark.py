from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional

from ..algorithms.scheduling import RandomScheduler, SchedulingResult
from ..framework.configuration import AppConfig
from ..framework.dataset_loading import DatasetLoader
from .metrics_collector import MetricsSnapshot


@dataclass(frozen=True)
class RandomScheduleRunResult:
    run_index: int
    seed: int
    assignment: SchedulingResult
    metrics: MetricsSnapshot
    energy_total: float
    duration: float
    sla_objective: float
    fitness: float
    uncertainty_spread: float


@dataclass(frozen=True)
class RandomBenchmarkStatistics:
    sample_count: int
    runs: List[RandomScheduleRunResult]
    energy_max_rand: float
    duration_max_rand: float
    uncertainty_max_rand: float
    sla_objective_max_rand: float


@dataclass(frozen=True)
class RandomBenchmarkNormalization:
    """Per-epoch normalization maxima derived from random benchmark schedules."""

    energy_norm_max: float
    sla_norm_max: float
    sample_count: int


class RandomScheduleBenchmarkRunner:
    """Runs random schedules and computes paper-style normalization benchmarks."""

    def __init__(
        self,
        dataset_loader: Optional[DatasetLoader] = None,
        random_seed: int = 0,
        uncertainty_repeats: int = 5,
        uncertainty_noise_std: float = 0.0,
    ) -> None:
        if uncertainty_repeats <= 0:
            raise ValueError("uncertainty_repeats must be > 0")
        if uncertainty_noise_std < 0:
            raise ValueError("uncertainty_noise_std must be >= 0")

        self._dataset_loader = dataset_loader or DatasetLoader()
        self._random_seed = random_seed
        self._uncertainty_repeats = uncertainty_repeats
        self._uncertainty_noise_std = uncertainty_noise_std

    def run(
        self,
        dataset_source: Dict[str, Any] | str,
        k: int = 100,
        app_config: AppConfig | None = None,
    ) -> RandomBenchmarkStatistics:
        if k <= 0:
            raise ValueError("k must be > 0")

        from ..framework.orchestrator import SimulationOrchestrator

        dataset = self._dataset_loader.load(dataset_source)
        rng = random.Random(self._random_seed)

        runs: List[RandomScheduleRunResult] = []

        for run_index in range(k):
            schedule_seed = rng.randint(0, 2**31 - 1)
            scheduler = RandomScheduler(seed=schedule_seed)
            orchestrator = SimulationOrchestrator(scheduler=scheduler, dataset_loader=self._dataset_loader)
            state = orchestrator.run(
                dataset,
                app_config=app_config,
                use_random_benchmark=False,
            )
            runtime = state.get("orchestrator_runtime")
            metrics: MetricsSnapshot = runtime.metrics
            metrics_collector = state.get("metrics_collector")
            xi = metrics_collector.fitness_parameters.xi

            sla_objective = metrics.sla.aggregate_penalty + (xi * metrics.fairness.combined_fairness)
            uncertainty = self._uncertainty_spread(metrics.fitness, rng)

            runs.append(
                RandomScheduleRunResult(
                    run_index=run_index,
                    seed=schedule_seed,
                    assignment=runtime.assignment,
                    metrics=metrics,
                    energy_total=metrics.energy.total_energy,
                    duration=metrics.makespan,
                    sla_objective=sla_objective,
                    fitness=metrics.fitness,
                    uncertainty_spread=uncertainty,
                )
            )

        energy_max = max(run.energy_total for run in runs)
        duration_max = max(run.duration for run in runs)
        uncertainty_max = max(run.uncertainty_spread for run in runs)
        sla_objective_max = max(run.sla_objective for run in runs)

        return RandomBenchmarkStatistics(
            sample_count=k,
            runs=runs,
            energy_max_rand=energy_max,
            duration_max_rand=duration_max,
            uncertainty_max_rand=uncertainty_max,
            sla_objective_max_rand=sla_objective_max,
        )

    def _uncertainty_spread(self, base_fitness: float, rng: random.Random) -> float:
        noisy_values: List[float] = []
        for _ in range(self._uncertainty_repeats):
            noise = rng.gauss(0.0, self._uncertainty_noise_std)
            noisy_values.append(base_fitness + noise)

        return max(noisy_values) - min(noisy_values)


class RandomBenchmarkNormalizer:
    """Computes per-epoch normalization maxima from random feasible schedules."""

    def __init__(
        self,
        dataset_loader: Optional[DatasetLoader] = None,
        random_seed: int = 0,
    ) -> None:
        self._runner = RandomScheduleBenchmarkRunner(
            dataset_loader=dataset_loader,
            random_seed=random_seed,
        )

    def compute(
        self,
        dataset_source: Dict[str, Any] | str,
        app_config: AppConfig,
        sample_count: int,
    ) -> RandomBenchmarkNormalization:
        stats = self._runner.run(
            dataset_source=dataset_source,
            k=sample_count,
            app_config=app_config,
        )
        return RandomBenchmarkNormalization(
            energy_norm_max=max(stats.energy_max_rand, 1e-12),
            sla_norm_max=max(stats.sla_objective_max_rand, 1e-12),
            sample_count=stats.sample_count,
        )
