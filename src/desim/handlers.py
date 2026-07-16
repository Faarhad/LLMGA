from dataclasses import dataclass
from typing import Protocol

from .dispatcher import EventDispatcher
from .event import Event, EventHandler
from .state import SimulationState


class NamedEventHandler(Protocol):
    """Protocol for reusable event handlers bound to an event name."""

    event_name: str

    def handle(self, state: SimulationState, event: Event) -> None:
        ...


@dataclass(frozen=True)
class CallbackEventHandler:
    """Small adapter to register a callback as a named event handler."""

    event_name: str
    callback: EventHandler

    def __post_init__(self) -> None:
        if not self.event_name:
            raise ValueError("event_name must be non-empty")

    def handle(self, state: SimulationState, event: Event) -> None:
        self.callback(state, event)

    def register(self, dispatcher: EventDispatcher) -> None:
        dispatcher.register(self.event_name, self.handle)
