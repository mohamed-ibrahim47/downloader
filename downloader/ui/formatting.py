"""Numbers as people read them: sizes, speeds, time left."""
from __future__ import annotations


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
