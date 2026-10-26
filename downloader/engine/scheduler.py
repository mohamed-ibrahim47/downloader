"""Start the queue at a set time (and optionally stop it) on chosen days."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Literal


@dataclass
class Schedule:
    enabled: bool = False
    start: str = "02:00"
    stop: str | None = None
    days: list[int] = field(default_factory=lambda: list(range(7)))  # Monday = 0


def _crossed(hhmm: str, days: list[int], prev: datetime, now: datetime) -> bool:
    h, m = (int(x) for x in hhmm.split(":"))
    day = prev.date()
    while day <= now.date():
        moment = datetime.combine(day, time(h, m))
        if prev < moment <= now and moment.weekday() in days:
            return True
        day += timedelta(days=1)
    return False


def due_action(s: Schedule, prev: datetime, now: datetime) -> Literal["start", "stop"] | None:
    """What the scheduler should do for the time that passed between `prev` and `now`."""
    if not s.enabled:
        return None
    if _crossed(s.start, s.days, prev, now):
        return "start"
    if s.stop and _crossed(s.stop, s.days, prev, now):
        return "stop"
    return None
