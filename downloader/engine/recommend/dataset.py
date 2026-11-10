"""The learning log turned into training rows: one per video shown, with a label and a weight (design C2).

| after it was shown | label | weight         |
|--------------------|-------|----------------|
| downloaded         | 1     | 1.0            |
| dismissed          | 0     | 1.0            |
| nothing            | 0     | IGNORED_WEIGHT |

Scrolling past is a weak "no": maybe the owner didn't like it, maybe they just didn't look.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .features import from_logged

IGNORED_WEIGHT = 0.3
MAX_ROWS = 5000  # the newest rows only: training stays fast, and old taste fades
_ACTIONS = {"downloaded": 1, "dismissed": 0}


@dataclass
class Row:
    video_id: str
    x: list[float]
    label: int
    weight: float
    refresh: float  # which list it was shown in
    at: float


def rows(events: list[tuple], limit: int = MAX_ROWS, open_refresh: float | None = None) -> list[Row]:
    """`events`: (id, video_id, event, features_json, at, refresh) tuples in id order, as
    `RecLog.events()` returns them.

    - A download or dismissal belongs to the video's last showing before it; earlier showings count as
      scrolled past.
    - Untouched rows of the list still on the page are left out: the owner may still act on them. That
      list is `open_refresh` (the latest refresh), or by default the newest list shown.
    - Rows logged before 1.2.1 (no refresh) are left out: they lack the search kind.
    """
    shown = [e for e in events if e[2] == "shown" and e[5] is not None and e[3]]
    if not shown:
        return []
    newest = open_refresh if open_refresh is not None else max(e[5] for e in shown)
    # For each video, its actions in order: (event id, label).
    actions: dict[str, list[tuple[int, int]]] = {}
    for eid, vid, event, *_ in events:
        if event in _ACTIONS:
            actions.setdefault(vid, []).append((eid, _ACTIONS[event]))
    # For each video, the ids of its showings, to find "the last showing before an action".
    showings: dict[str, list[int]] = {}
    for e in shown:
        showings.setdefault(e[1], []).append(e[0])

    out = []
    for eid, vid, _event, features, at, refresh in shown:
        nxt = next(((aid, label) for aid, label in actions.get(vid, ()) if aid > eid), None)
        if nxt is not None and not any(eid < sid < nxt[0] for sid in showings[vid]):
            label, weight = nxt[1], 1.0  # the last showing before the action
        elif refresh == newest and nxt is None:
            continue  # still on the page: no verdict yet
        else:
            label, weight = 0, IGNORED_WEIGHT
        out.append(Row(vid, from_logged(json.loads(features)), label, weight, refresh, at))
    return out[-limit:]
