import heapq
import itertools
from typing import Any, List, Optional, Tuple

from .event import Event


class PriorityQueue:
    """Stable min-priority queue using heapq."""

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, Any]] = []
        self._counter = itertools.count()

    def push(self, priority: float, item: Any) -> None:
        heapq.heappush(self._heap, (priority, next(self._counter), item))

    def pop(self) -> Any:
        if not self._heap:
            raise IndexError("pop from empty priority queue")
        _, _, item = heapq.heappop(self._heap)
        return item

    def peek(self) -> Any:
        if not self._heap:
            raise IndexError("peek from empty priority queue")
        return self._heap[0][2]

    def clear(self) -> None:
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0


class EventQueue:
    """Queue for simulation events ordered by event time."""

    def __init__(self) -> None:
        self._pq = PriorityQueue()

    def schedule(self, event: Event) -> None:
        self._pq.push(event.time, event)

    def pop_next(self) -> Event:
        return self._pq.pop()

    def peek_next(self) -> Event:
        return self._pq.peek()

    def clear(self) -> None:
        self._pq.clear()

    def __len__(self) -> int:
        return len(self._pq)

    def is_empty(self) -> bool:
        return self._pq.is_empty()

    def next_time(self) -> Optional[float]:
        if self._pq.is_empty():
            return None
        return self.peek_next().time
