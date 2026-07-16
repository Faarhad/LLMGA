from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class SimulationLifecycle(str, Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"


@dataclass
class SimulationState:
    """Mutable simulation state container."""

    lifecycle: SimulationLifecycle = SimulationLifecycle.CREATED
    data: Dict[str, Any] = field(default_factory=dict)
    events_processed: int = 0

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def reset(self) -> None:
        self.lifecycle = SimulationLifecycle.CREATED
        self.data.clear()
        self.events_processed = 0
