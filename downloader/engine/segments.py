"""Splits a file into byte ranges that separate connections download at the same time."""
from __future__ import annotations

from .models import Segment


def plan_segments(size: int | None, connections: int, resumable: bool,
                  min_part: int = 1_048_576) -> list[Segment]:
    if size is None:
        return [Segment(0, 0, -1)]
    if size <= 0:
        return []  # nothing to fetch: an empty file
    if not resumable:
        return [Segment(0, 0, size - 1)]
    count = min(connections, max(1, size // min_part))
    part = size // count
    segments = []
    for i in range(count):
        start = i * part
        end = size - 1 if i == count - 1 else start + part - 1
        segments.append(Segment(i, start, end))
    return segments
