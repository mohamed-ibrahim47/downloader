"""Who the owner is, as the recommender sees it: their video downloads, fading with age (spec B4)."""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..models import Download, Kind
from .text import tokenize

MIN_DOWNLOADS = 3  # fewer than this and there is no profile yet (cold start)
HALF_LIFE_DAYS = 30  # a download this old counts half as much as one from today
DISMISS_WEIGHT = -0.5  # a dismissed suggestion pulls its words down by this much

_DAY = 86400
_KINDS = (Kind.VIDEO, Kind.PLAYLIST_ITEM)


@dataclass
class ProfileDoc:
    video_id: str | None
    title: str
    tokens: list[str]
    weight: float


@dataclass
class Dismissed:
    video_id: str
    title: str
    channel: str | None


@dataclass
class Profile:
    docs: list[ProfileDoc]
    channels: dict[str, float]  # display name -> affinity, the top channel = 1.0

    @property
    def ready(self) -> bool:
        return sum(1 for d in self.docs if d.weight > 0) >= MIN_DOWNLOADS


def _tags(text: str | None) -> list[str]:
    """The stored JSON tag list; anything that isn't a list of strings counts as no tags."""
    try:
        tags = json.loads(text) if text else []
    except ValueError:
        return []
    return [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []


def _recency(added_at: float, now: float) -> float:
    age_days = max(0.0, (now - added_at) / _DAY)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def build_profile(downloads: list[Download], dismissed: list[Dismissed], now: float) -> Profile:
    docs: list[ProfileDoc] = []
    totals: dict[str, float] = {}
    for d in downloads:
        if d.kind not in _KINDS or not d.title:
            continue
        text = f"{d.title} {d.playlist_title}" if d.playlist_title else d.title
        weight = _recency(d.added_at, now)
        docs.append(ProfileDoc(d.video_id, d.title, tokenize(text, _tags(d.tags), d.channel), weight))
        if d.channel:
            totals[d.channel] = totals.get(d.channel, 0.0) + weight
    for x in dismissed:
        docs.append(ProfileDoc(x.video_id, x.title, tokenize(x.title, channel=x.channel), DISMISS_WEIGHT))
    top = max(totals.values(), default=0.0)
    channels = {name: total / top for name, total in totals.items()} if top > 0 else {}
    return Profile(docs, channels)
