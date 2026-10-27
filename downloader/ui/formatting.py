"""Numbers as people read them: sizes, speeds, time left. Plus plain-text helpers for Qt."""
from __future__ import annotations

import html
import time


def plain_tip(text: str) -> str:
    """Tooltip text that Qt can never read as HTML (names and errors come from servers)."""
    return f"<p style='white-space:pre-wrap'>{html.escape(text)}</p>"


def fmt_duration(seconds: int | None) -> str:
    if not seconds:
        return ""
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m} min {s} s" if s else f"{m} min"
    return f"{seconds // 3600} h {(seconds % 3600) // 60} min"


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return ""
    if n < 1000:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1000
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}"
    return ""


def fmt_speed(bps: float | None) -> str:
    return f"{fmt_bytes(bps or 0)}/s"


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(seconds)
    if s < 60:
        return f"about {max(s, 1)} s left"
    if s < 3600:
        return f"about {s // 60} min left"
    return f"about {s // 3600} h {(s % 3600) // 60} min left"


def fmt_day(ts: float, now: float | None = None) -> str:
    """"26 Oct", with the year only when it isn't this year's."""
    day = time.localtime(ts)
    this_year = time.localtime(now if now is not None else time.time()).tm_year
    text = f"{day.tm_mday} {time.strftime('%b', day)}"
    return text if day.tm_year == this_year else f"{text} {day.tm_year}"
