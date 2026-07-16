from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class SimulationConfig:
    until: Optional[float] = None
    max_events: Optional[int] = None


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "random"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnergyConfig:
    alpha_share: float = 1.0
    coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.alpha_share < 0 or self.alpha_share > 1:
            raise ConfigurationError("energy.alpha_share must be in [0, 1]")


@dataclass(frozen=True)
class MetricsConfig:
    sla_lambda: float = 1.0
    sla_theta: float = 1.0
    sla_eta_max: float = 2.0
    fairness_omega_1: float = 0.5
    fairness_omega_2: float = 0.5
    fairness_mu: float = 1.0
    fitness_w_energy: float = 0.5
    fitness_w_sla: float = 0.5
    fitness_xi: float = 1.0
    fitness_energy_norm_max: float = 1.0
    fitness_sla_norm_max: float = 1.0

    def __post_init__(self) -> None:
        if self.sla_lambda <= 0 or self.sla_theta <= 0 or self.sla_eta_max <= 0:
            raise ConfigurationError("metrics.sla parameters must be > 0")
        if self.fairness_omega_1 < 0 or self.fairness_omega_2 < 0:
            raise ConfigurationError("metrics.fairness omegas must be >= 0")
        if abs((self.fairness_omega_1 + self.fairness_omega_2) - 1.0) > 1e-9:
            raise ConfigurationError("metrics.fairness omega_1 + omega_2 must equal 1")
        if self.fairness_mu < 0:
            raise ConfigurationError("metrics.fairness mu must be >= 0")
        if self.fitness_w_energy < 0 or self.fitness_w_sla < 0:
            raise ConfigurationError("metrics.fitness weights must be >= 0")
        if abs((self.fitness_w_energy + self.fitness_w_sla) - 1.0) > 1e-9:
            raise ConfigurationError("metrics.fitness w_energy + w_sla must equal 1")
        if self.fitness_xi < 0:
            raise ConfigurationError("metrics.fitness xi must be >= 0")
        if self.fitness_energy_norm_max <= 0 or self.fitness_sla_norm_max <= 0:
            raise ConfigurationError("metrics.fitness normalization maxima must be > 0")


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    format: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ConfigurationError("dataset.source must be non-empty")


@dataclass(frozen=True)
class RandomSeedsConfig:
    global_seed: Optional[int] = None
    scheduler_seed: Optional[int] = None


@dataclass(frozen=True)
class PluginConfig:
    name: str
    enabled: bool = True
    entrypoint: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("plugin.name must be non-empty")


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig
    scheduler: SchedulerConfig
    energy: EnergyConfig
    metrics: MetricsConfig
    dataset: DatasetConfig
    random_seeds: RandomSeedsConfig = field(default_factory=RandomSeedsConfig)
    plugins: List[PluginConfig] = field(default_factory=list)


class AppConfigLoader:
    """Loads application configuration from YAML or dictionary."""

    def load_from_yaml(self, path: str | Path) -> AppConfig:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ConfigurationError("PyYAML is required for YAML configuration") from exc

        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ConfigurationError("YAML root must be a mapping")
        return self.load_from_dict(raw)

    def load_from_dict(self, raw: Dict[str, Any]) -> AppConfig:
        simulation_raw = self._as_mapping(raw.get("simulation", {}), "simulation")
        scheduler_raw = self._as_mapping(raw.get("scheduler", {}), "scheduler")
        energy_raw = self._as_mapping(raw.get("energy", {}), "energy")
        metrics_raw = self._as_mapping(raw.get("metrics", {}), "metrics")

        if "dataset" not in raw:
            raise ConfigurationError("dataset section is required")
        dataset_raw = self._as_mapping(raw["dataset"], "dataset")

        random_seeds_raw = self._as_mapping(raw.get("random_seeds", {}), "random_seeds")
        plugins_raw = raw.get("plugins", [])
        if not isinstance(plugins_raw, list):
            raise ConfigurationError("plugins must be a list")

        plugins = [self._parse_plugin(item, i) for i, item in enumerate(plugins_raw)]

        return AppConfig(
            simulation=SimulationConfig(
                until=self._optional_float(simulation_raw.get("until")),
                max_events=self._optional_int(simulation_raw.get("max_events")),
            ),
            scheduler=SchedulerConfig(
                name=str(scheduler_raw.get("name", "random")),
                options=self._as_mapping(scheduler_raw.get("options", {}), "scheduler.options"),
            ),
            energy=EnergyConfig(
                alpha_share=float(energy_raw.get("alpha_share", 1.0)),
                coefficients=self._as_mapping(energy_raw.get("coefficients", {}), "energy.coefficients"),
            ),
            metrics=MetricsConfig(
                sla_lambda=float(metrics_raw.get("sla_lambda", 1.0)),
                sla_theta=float(metrics_raw.get("sla_theta", 1.0)),
                sla_eta_max=float(metrics_raw.get("sla_eta_max", 2.0)),
                fairness_omega_1=float(metrics_raw.get("fairness_omega_1", 0.5)),
                fairness_omega_2=float(metrics_raw.get("fairness_omega_2", 0.5)),
                fairness_mu=float(metrics_raw.get("fairness_mu", 1.0)),
                fitness_w_energy=float(metrics_raw.get("fitness_w_energy", 0.5)),
                fitness_w_sla=float(metrics_raw.get("fitness_w_sla", 0.5)),
                fitness_xi=float(metrics_raw.get("fitness_xi", 1.0)),
                fitness_energy_norm_max=float(metrics_raw.get("fitness_energy_norm_max", 1.0)),
                fitness_sla_norm_max=float(metrics_raw.get("fitness_sla_norm_max", 1.0)),
            ),
            dataset=DatasetConfig(
                source=str(dataset_raw.get("source", "")),
                format=str(dataset_raw["format"]) if "format" in dataset_raw and dataset_raw["format"] is not None else None,
            ),
            random_seeds=RandomSeedsConfig(
                global_seed=self._optional_int(random_seeds_raw.get("global_seed")),
                scheduler_seed=self._optional_int(random_seeds_raw.get("scheduler_seed")),
            ),
            plugins=plugins,
        )

    @staticmethod
    def _as_mapping(value: Any, name: str) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ConfigurationError(f"{name} must be a mapping")
        return dict(value)

    @staticmethod
    def _parse_plugin(value: Any, index: int) -> PluginConfig:
        if not isinstance(value, dict):
            raise ConfigurationError(f"plugins[{index}] must be a mapping")
        return PluginConfig(
            name=str(value.get("name", "")),
            enabled=bool(value.get("enabled", True)),
            entrypoint=str(value["entrypoint"]) if "entrypoint" in value and value["entrypoint"] is not None else None,
            options=dict(value.get("options", {})) if isinstance(value.get("options", {}), dict) else {},
        )

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value)
