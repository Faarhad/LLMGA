from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, List, Optional, Protocol
import csv
import json
import logging
import math
import random

from .metrics_collector import MetricsSnapshot


class ExperimentRunner(Protocol):
    """Future GA experiments should implement this interface."""

    def run(self, seed: int) -> MetricsSnapshot:
        ...


@dataclass(frozen=True)
class ExperimentRunRecord:
    run_index: int
    seed: int
    metrics: MetricsSnapshot
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def makespan(self) -> float:
        return self.metrics.makespan

    @property
    def throughput(self) -> float:
        return self.metrics.throughput

    @property
    def energy(self) -> float:
        return self.metrics.energy.total_energy

    @property
    def waiting_time(self) -> float:
        return self.metrics.average_waiting_time

    @property
    def response_time(self) -> float:
        return self.metrics.average_response_time

    @property
    def sla_penalty(self) -> float:
        return self.metrics.sla.aggregate_penalty

    @property
    def fairness(self) -> float:
        return self.metrics.fairness.combined_fairness

    @property
    def utilization(self) -> float:
        return self.metrics.cpu_occupancy

    @property
    def fitness(self) -> float:
        return self.metrics.fitness

    @property
    def uncertainty(self) -> float:
        return self.metrics.fairness.exponential_variance


@dataclass(frozen=True)
class ExperimentSummary:
    run_count: int
    seeds: List[int]
    energy_mean: float
    energy_std: float
    makespan_mean: float
    makespan_std: float
    throughput_mean: float
    throughput_std: float
    waiting_time_mean: float
    waiting_time_std: float
    response_time_mean: float
    response_time_std: float
    sla_penalty_mean: float
    sla_penalty_std: float
    fairness_mean: float
    fairness_std: float
    utilization_mean: float
    utilization_std: float
    fitness_mean: float
    fitness_std: float
    energy_max: float
    makespan_max: float
    throughput_max: float
    waiting_time_max: float
    response_time_max: float
    sla_penalty_max: float
    fairness_max: float
    utilization_max: float
    fitness_max: float


@dataclass(frozen=True)
class PlotArtifact:
    name: str
    path: Path


class ExperimentManager:
    """Runs repeated experiments, stores results, exports summaries and plots."""

    def __init__(
        self,
        runner: ExperimentRunner,
        output_dir: str | Path,
        base_seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.runner = runner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_seed = base_seed
        self.logger = logger or logging.getLogger(__name__)
        self.runs: List[ExperimentRunRecord] = []

    def run(self, run_count: int = 1) -> List[ExperimentRunRecord]:
        if run_count <= 0:
            raise ValueError("run_count must be > 0")

        self.logger.info("starting experiment batch", extra={"run_count": run_count, "base_seed": self.base_seed})
        rng = random.Random(self.base_seed)
        self.runs = []

        for index in range(run_count):
            seed = rng.randint(0, 2**31 - 1)
            self.logger.info("starting run", extra={"run_index": index, "seed": seed})
            metrics = self.runner.run(seed)
            self.runs.append(ExperimentRunRecord(run_index=index, seed=seed, metrics=metrics))
            self.logger.info(
                "finished run",
                extra={
                    "run_index": index,
                    "seed": seed,
                    "fitness": metrics.fitness,
                    "energy": metrics.energy.total_energy,
                    "makespan": metrics.makespan,
                },
            )

        self.logger.info("experiment batch complete", extra={"run_count": len(self.runs)})
        return list(self.runs)

    def summarize(self) -> ExperimentSummary:
        if not self.runs:
            raise ValueError("no runs available; call run() first")

        energy = [r.energy for r in self.runs]
        makespan = [r.makespan for r in self.runs]
        throughput = [r.throughput for r in self.runs]
        waiting_time = [r.waiting_time for r in self.runs]
        response_time = [r.response_time for r in self.runs]
        sla_penalty = [r.sla_penalty for r in self.runs]
        fairness = [r.fairness for r in self.runs]
        utilization = [r.utilization for r in self.runs]
        fitness = [r.fitness for r in self.runs]

        def safe_std(values: List[float]) -> float:
            return pstdev(values) if len(values) > 1 else 0.0

        return ExperimentSummary(
            run_count=len(self.runs),
            seeds=[r.seed for r in self.runs],
            energy_mean=mean(energy),
            energy_std=safe_std(energy),
            makespan_mean=mean(makespan),
            makespan_std=safe_std(makespan),
            throughput_mean=mean(throughput),
            throughput_std=safe_std(throughput),
            waiting_time_mean=mean(waiting_time),
            waiting_time_std=safe_std(waiting_time),
            response_time_mean=mean(response_time),
            response_time_std=safe_std(response_time),
            sla_penalty_mean=mean(sla_penalty),
            sla_penalty_std=safe_std(sla_penalty),
            fairness_mean=mean(fairness),
            fairness_std=safe_std(fairness),
            utilization_mean=mean(utilization),
            utilization_std=safe_std(utilization),
            fitness_mean=mean(fitness),
            fitness_std=safe_std(fitness),
            energy_max=max(energy),
            makespan_max=max(makespan),
            throughput_max=max(throughput),
            waiting_time_max=max(waiting_time),
            response_time_max=max(response_time),
            sla_penalty_max=max(sla_penalty),
            fairness_max=max(fairness),
            utilization_max=max(utilization),
            fitness_max=max(fitness),
        )

    def save_runs_json(self, file_name: str = "runs.json") -> Path:
        path = self.output_dir / file_name
        with path.open("w", encoding="utf-8") as f:
            json.dump([self._run_to_dict(run) for run in self.runs], f, indent=2)
        return path

    def save_summary_json(self, file_name: str = "summary.json") -> Path:
        path = self.output_dir / file_name
        summary = self.summarize()
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary.__dict__, f, indent=2)
        return path

    def export_csv(self, file_name: str = "runs.csv") -> Path:
        path = self.output_dir / file_name
        fieldnames = [
            "run_index",
            "seed",
            "makespan",
            "throughput",
            "energy",
            "waiting_time",
            "response_time",
            "sla_penalty",
            "fairness",
            "utilization",
            "fitness",
            "uncertainty",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for run in self.runs:
                writer.writerow(
                    {
                        "run_index": run.run_index,
                        "seed": run.seed,
                        "makespan": run.makespan,
                        "throughput": run.throughput,
                        "energy": run.energy,
                        "waiting_time": run.waiting_time,
                        "response_time": run.response_time,
                        "sla_penalty": run.sla_penalty,
                        "fairness": run.fairness,
                        "utilization": run.utilization,
                        "fitness": run.fitness,
                        "uncertainty": run.uncertainty,
                    }
                )
        return path

    def export_plots(self) -> List[PlotArtifact]:
        artifacts = [
            self._write_svg_plot("fitness.svg", "Fitness", [r.fitness for r in self.runs]),
            self._write_svg_plot("energy.svg", "Energy", [r.energy for r in self.runs]),
            self._write_svg_plot("makespan.svg", "Makespan", [r.makespan for r in self.runs]),
            self._write_svg_plot("sla_penalty.svg", "SLA Penalty", [r.sla_penalty for r in self.runs]),
            self._write_svg_plot("fairness.svg", "Fairness", [r.fairness for r in self.runs]),
        ]
        return artifacts

    def _run_to_dict(self, run: ExperimentRunRecord) -> Dict[str, object]:
        return {
            "run_index": run.run_index,
            "seed": run.seed,
            "metrics": {
                "makespan": run.makespan,
                "throughput": run.throughput,
                "energy": run.energy,
                "waiting_time": run.waiting_time,
                "response_time": run.response_time,
                "sla_penalty": run.sla_penalty,
                "fairness": run.fairness,
                "utilization": run.utilization,
                "fitness": run.fitness,
                "uncertainty": run.uncertainty,
            },
        }

    def _write_svg_plot(self, file_name: str, title: str, values: List[float]) -> PlotArtifact:
        path = self.output_dir / file_name
        svg = self._build_svg(title, values)
        path.write_text(svg, encoding="utf-8")
        return PlotArtifact(name=title, path=path)

    @staticmethod
    def _build_svg(title: str, values: List[float]) -> str:
        width = 900
        height = 300
        margin = 40
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin

        if not values:
            values = [0.0]

        min_value = min(values)
        max_value = max(values)
        if math.isclose(max_value, min_value):
            max_value = min_value + 1.0

        def scale_x(index: int) -> float:
            if len(values) == 1:
                return margin + plot_width / 2
            return margin + (index / (len(values) - 1)) * plot_width

        def scale_y(value: float) -> float:
            return margin + plot_height - ((value - min_value) / (max_value - min_value)) * plot_height

        points = " ".join(f"{scale_x(i):.1f},{scale_y(v):.1f}" for i, v in enumerate(values))
        bars = []
        if len(values) > 1:
            bar_width = plot_width / len(values) * 0.6
            for i, value in enumerate(values):
                x = scale_x(i) - bar_width / 2
                y = scale_y(value)
                h = margin + plot_height - y
                bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="#4e79a7" />')
        else:
            x = scale_x(0)
            y = scale_y(values[0])
            bars.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#4e79a7" />')

        return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\" />
  <text x=\"{margin}\" y=\"24\" font-family=\"Arial\" font-size=\"18\" fill=\"#111\">{title}</text>
  <line x1=\"{margin}\" y1=\"{margin + plot_height}\" x2=\"{margin + plot_width}\" y2=\"{margin + plot_height}\" stroke=\"#333\" />
  <line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{margin + plot_height}\" stroke=\"#333\" />
  {' '.join(bars)}
  <polyline points=\"{points}\" fill=\"none\" stroke=\"#e15759\" stroke-width=\"2\" />
</svg>"""
