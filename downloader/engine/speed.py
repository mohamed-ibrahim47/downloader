"""Measures download speed: the current rate, a minute of history for the sparkline, time left."""
from __future__ import annotations

import math
import threading
from collections import deque

RATE_WINDOW = 3.0


class SpeedMeter:
    def __init__(self, window: int = 60):
        self._window = window
        self._samples: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def add(self, nbytes: int, now: float) -> None:
        with self._lock:
            self._samples.append((now, nbytes))
            while self._samples and self._samples[0][0] <= now - self._window - 1:
                self._samples.popleft()

    def rate(self, now: float) -> float:
        with self._lock:
            total = sum(n for t, n in self._samples if now - RATE_WINDOW < t <= now)
        return total / RATE_WINDOW

    def history(self, now: float) -> list[float]:
        """Bytes per second for each of the last `window` whole seconds, oldest first."""
        last = math.floor(now)
        buckets = [0.0] * self._window
        with self._lock:
            for t, n in self._samples:
                i = self._window - 1 - (last - math.floor(t))
                if 0 <= i < self._window:
                    buckets[i] += n
        return buckets

    def eta(self, remaining: int | None, now: float) -> float | None:
        rate = self.rate(now)
        if remaining is None or rate <= 0:
            return None
        return remaining / rate
