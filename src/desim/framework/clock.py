class SimulationClock:
    """Monotonic simulation clock."""

    def __init__(self, start_time: float = 0.0) -> None:
        if start_time < 0:
            raise ValueError("start_time must be >= 0")
        self._time = float(start_time)

    @property
    def now(self) -> float:
        return self._time

    def advance_to(self, new_time: float) -> None:
        if new_time < self._time:
            raise ValueError("Simulation time cannot move backwards")
        self._time = float(new_time)

    def reset(self, start_time: float = 0.0) -> None:
        if start_time < 0:
            raise ValueError("start_time must be >= 0")
        self._time = float(start_time)
