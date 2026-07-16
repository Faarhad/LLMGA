from typing import Optional

from .clock import SimulationClock
from .dispatcher import EventDispatcher
from .event import Event
from .queue import EventQueue
from .state import SimulationLifecycle, SimulationState


class Simulation:
    """Discrete-event simulation engine."""

    def __init__(
        self,
        clock: Optional[SimulationClock] = None,
        event_queue: Optional[EventQueue] = None,
        dispatcher: Optional[EventDispatcher] = None,
        state: Optional[SimulationState] = None,
    ) -> None:
        self.clock = clock or SimulationClock()
        self.event_queue = event_queue or EventQueue()
        self.dispatcher = dispatcher or EventDispatcher()
        self.state = state or SimulationState()

    def initialize(self) -> None:
        if self.state.lifecycle != SimulationLifecycle.CREATED:
            raise RuntimeError("Simulation can only be initialized from CREATED")
        self.state.lifecycle = SimulationLifecycle.INITIALIZED

    def schedule(self, event: Event) -> None:
        self.event_queue.schedule(event)

    def schedule_at(self, time: float, name: str, payload: Optional[dict] = None) -> None:
        self.schedule(Event(time=time, name=name, payload=payload or {}))

    def step(self) -> bool:
        if self.event_queue.is_empty():
            return False

        event = self.event_queue.pop_next()
        self.clock.advance_to(event.time)
        self.dispatcher.dispatch(self.state, event)
        self.state.events_processed += 1
        return True

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> int:
        if self.state.lifecycle == SimulationLifecycle.CREATED:
            self.initialize()

        if self.state.lifecycle not in {
            SimulationLifecycle.INITIALIZED,
            SimulationLifecycle.RUNNING,
        }:
            raise RuntimeError("Simulation must be INITIALIZED or RUNNING to run")

        self.state.lifecycle = SimulationLifecycle.RUNNING
        processed_in_run = 0

        while not self.event_queue.is_empty():
            next_time = self.event_queue.next_time()

            if until is not None and next_time is not None and next_time > until:
                break

            if max_events is not None and processed_in_run >= max_events:
                break

            self.step()
            processed_in_run += 1

            if self.state.lifecycle == SimulationLifecycle.STOPPED:
                break

        if self.event_queue.is_empty() and self.state.lifecycle != SimulationLifecycle.STOPPED:
            self.state.lifecycle = SimulationLifecycle.COMPLETED
        elif self.state.lifecycle == SimulationLifecycle.RUNNING:
            self.state.lifecycle = SimulationLifecycle.INITIALIZED

        return processed_in_run

    def stop(self) -> None:
        self.state.lifecycle = SimulationLifecycle.STOPPED

    def reset(self) -> None:
        self.clock.reset()
        self.event_queue.clear()
        self.state.reset()
