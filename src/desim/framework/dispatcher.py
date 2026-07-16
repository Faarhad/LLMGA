from collections import defaultdict
from typing import DefaultDict, List

from .event import Event, EventHandler
from .state import SimulationState


class EventDispatcher:
    """Dispatches events to event-local and registered handlers."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[str, List[EventHandler]] = defaultdict(list)

    def register(self, event_name: str, handler: EventHandler) -> None:
        if not event_name:
            raise ValueError("event_name must be non-empty")
        self._handlers[event_name].append(handler)

    def dispatch(self, state: SimulationState, event: Event) -> None:
        if event.handler is not None:
            event.handler(state, event)

        for handler in self._handlers[event.name]:
            handler(state, event)
