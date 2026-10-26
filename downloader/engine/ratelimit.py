"""A shared speed limit for every connection of every download."""
from __future__ import annotations

import threading
import time


class TokenBucket:
    """Each byte costs a token; tokens refill at `rate_bps` per second, up to one second's worth.

    A caller that takes more than is available goes into debt and sleeps it off, so large
    chunks and many threads still average out to the limit. `None` means no limit.
    """

    def __init__(self, rate_bps: int | None = None):
        self._lock = threading.Lock()
        self._rate = rate_bps
        self._tokens = float(rate_bps or 0)
        self._last = time.monotonic()

    @property
    def rate(self) -> int | None:
        return self._rate

    def set_rate(self, rate_bps: int | None) -> None:
        with self._lock:
            self._rate = rate_bps
            self._tokens = min(self._tokens, float(rate_bps or 0))
            self._last = time.monotonic()

    def consume(self, n: int) -> None:
        with self._lock:
            if not self._rate:
                return
            now = time.monotonic()
            self._tokens = min(float(self._rate), self._tokens + (now - self._last) * self._rate)
            self._last = now
            self._tokens -= n
            wait = -self._tokens / self._rate if self._tokens < 0 else 0.0
        if wait:
            time.sleep(wait)
