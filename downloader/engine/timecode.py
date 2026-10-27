"""Times in a video as people type them (750, 12:30, 1:02:03, 12m30s) and as clips show them."""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

_COLON = re.compile(r"^(\d+)(?::(\d{1,2}))?(?::(\d{1,2}))?$")
_UNITS = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


def parse_time(text: str) -> int | None:
    """Seconds, or None if the text isn't a time."""
    text = text.strip().lower()
    if not text:
        return None
    m = _COLON.match(text)
    if m:
        parts = [int(p) for p in m.groups() if p is not None]
        if any(p > 59 for p in parts[1:]):
            return None
        total = 0
        for p in parts:
            total = total * 60 + p
        return total
    m = _UNITS.match(text)
    if m and any(m.groups()):
        h, mins, s = (int(p) if p else None for p in m.groups())
        # A smaller unit after a larger one must fit in it (1h75m isn't a time); alone it may not (90m).
        if mins is not None and h is not None and mins > 59:
            return None
        if s is not None and (h is not None or mins is not None) and s > 59:
            return None
        return (h or 0) * 3600 + (mins or 0) * 60 + (s or 0)
    return None


def fmt_clock(seconds: int) -> str:
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def start_from_url(url: str) -> int | None:
    """The start time a shared link carries: `t=` or `start=` in the query, or `#t=`."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    query = parse_qs(parts.query)
    fragment = parse_qs(parts.fragment)
    for value in (*query.get("t", []), *query.get("start", []), *fragment.get("t", [])):
        seconds = parse_time(value)
        if seconds is not None:
            return seconds
    return None


def clip_label(start: int, end: int) -> str:
    return f"Clip {fmt_clock(start)}–{fmt_clock(end)}"


def clip_suffix(start: int, end: int) -> str:
    """Added to a clip's file name; dashes because Windows names can't hold a colon."""
    return f" ({fmt_clock(start).replace(':', '-')} to {fmt_clock(end).replace(':', '-')})"
