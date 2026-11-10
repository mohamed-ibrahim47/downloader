"""The inputs v2's model sees for one video (design C3), as numbers in a fixed order.

The same inputs are built two ways: from a `shown` row's logged JSON (training), and from a fresh
`Suggestion` (ranking). When ranking, every video is put at rank 1 and not the discovery pick, so the
position inputs add the same amount to every score and can't change the order (position bias, C3).
"""
from __future__ import annotations

import math

from .rank import Suggestion

NAMES = ["similarity", "affinity", "shared", "log_duration", "kind_channel", "kind_newest", "log_rank",
         "discovery"]


def _vector(similarity: float, affinity: float, shared: int, duration: int | None, kind: str | None,
            rank: int, discovery: bool) -> list[float]:
    return [
        float(similarity),
        float(affinity),
        float(shared),
        math.log(1 + (duration or 0) / 60),  # minutes, squeezed: 10 vs 20 min matters more than 90 vs 100
        1.0 if kind == "channel" else 0.0,
        1.0 if kind == "newest" else 0.0,  # "words" (and unknown) is the base case: both 0
        math.log(max(rank, 1)),
        1.0 if discovery else 0.0,
    ]


def from_logged(f: dict) -> list[float]:
    """A `shown` row's features JSON (as `service._features` writes it) → the model's inputs."""
    return _vector(f.get("similarity", 0.0), f.get("affinity", 0.0), f.get("shared", 0), f.get("duration"),
                   f.get("kind"), f.get("rank", 1), bool(f.get("discovery")))


def from_suggestion(s: Suggestion, affinity: float) -> list[float]:
    """A candidate about to be ranked → the model's inputs, at rank 1 and not the discovery pick."""
    return _vector(s.similarity, affinity, len(s.shared), s.candidate.duration, s.candidate.kind, 1, False)


def at_top(x: list[float]) -> list[float]:
    """A logged row's inputs as ranking sees them: at rank 1 (log 1 = 0) and not the discovery pick."""
    return x[:6] + [0.0, 0.0]
