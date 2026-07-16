from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


EventHandler = Callable[["SimulationState", "Event"], None]


@dataclass(frozen=True)
class Event:
    """A timestamped simulation event."""

    time: float
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[EventHandler] = None

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError("event time must be >= 0")
        if not self.name:
            raise ValueError("event name must be non-empty")
