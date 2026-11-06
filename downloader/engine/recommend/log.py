"""The recommender's memory in SQLite: cached search results, what happened, blocked channels (spec B2).

`rec_events` is the learning log. Every suggestion shown is written with its score parts, so a later
version can train on "shown, then downloaded or not". It is never deleted in v1.
"""
from __future__ import annotations

import json

from ..store import Store
from .candidates import Candidate
from .profile import Dismissed

_LAST_REFRESH = "rec_last_refresh"  # a settings key: a refresh that found nothing still counts

_TABLES = (
    "CREATE TABLE IF NOT EXISTS rec_candidates (video_id TEXT PRIMARY KEY, url TEXT, title TEXT, "
    "channel TEXT, duration INTEGER, query TEXT, found_at REAL)",
    "CREATE TABLE IF NOT EXISTS rec_events (id INTEGER PRIMARY KEY, video_id TEXT, event TEXT, title TEXT, "
    "channel TEXT, score REAL, rank INTEGER, features TEXT, at REAL)",
    "CREATE TABLE IF NOT EXISTS rec_blocked_channels (channel TEXT PRIMARY KEY, at REAL)",
)

_INSERT_EVENT = ("INSERT INTO rec_events (video_id, event, title, channel, score, rank, features, at) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?)")


class RecLog:
    def __init__(self, store: Store):
        self.store = store
        store.execute_many([(sql, ()) for sql in _TABLES])

    # ---- the candidate cache -----------------------------------------------------

    def save_candidates(self, cands: list[Candidate], now: float) -> None:
        """Replace the cache in one transaction. The same video from two queries: the first one wins."""
        rows = [("INSERT OR IGNORE INTO rec_candidates VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (c.video_id, c.url, c.title, c.channel, c.duration, c.query, now)) for c in cands]
        self.store.execute_many([("DELETE FROM rec_candidates", ()), *rows])
        self.store.set_setting(_LAST_REFRESH, now)

    def candidates(self) -> list[Candidate]:
        rows = self.store.query("SELECT video_id, url, title, channel, duration, query FROM rec_candidates "
                                "ORDER BY rowid")
        return [Candidate(*row) for row in rows]

    def last_refresh(self) -> float | None:
        return self.store.get_setting(_LAST_REFRESH)

    # ---- events ------------------------------------------------------------------

    def record(self, event: str, cand: Candidate, now: float, score: float | None = None,
               rank: int | None = None, features: dict | None = None) -> None:
        self.record_many([(event, cand, score, rank, features)], now)

    def record_many(self, events: list[tuple], now: float) -> None:
        """(event, candidate, score, rank, features) tuples, written together."""
        self.store.execute_many([
            (_INSERT_EVENT, (c.video_id, event, c.title, c.channel, score, rank,
                             json.dumps(features) if features is not None else None, now))
            for event, c, score, rank, features in events])

    def dismissed(self) -> list[Dismissed]:
        rows = self.store.query("SELECT video_id, title, channel FROM rec_events WHERE event = 'dismissed' "
                                "GROUP BY video_id ORDER BY MIN(id)")
        return [Dismissed(*row) for row in rows]

    def mark_downloaded(self, downloaded_ids: set[str], now: float) -> None:
        """A shown suggestion that is now among the downloads gets one `downloaded` event, ever."""
        rows = self.store.query(
            "SELECT video_id, title, channel FROM rec_events WHERE event = 'shown' AND video_id NOT IN "
            "(SELECT video_id FROM rec_events WHERE event = 'downloaded') GROUP BY video_id")
        new = [(vid, title, channel) for vid, title, channel in rows if vid in downloaded_ids]
        if new:
            self.store.execute_many([(_INSERT_EVENT, (vid, "downloaded", title, channel, None, None, None, now))
                                     for vid, title, channel in new])

    # ---- blocked channels --------------------------------------------------------

    def block_channel(self, name: str, now: float) -> None:
        self.store.execute_many([
            ("INSERT OR REPLACE INTO rec_blocked_channels VALUES (?, ?)", (name, now)),
            (_INSERT_EVENT, (None, "blocked_channel", None, name, None, None, None, now)),
        ])

    def unblock_channel(self, name: str) -> None:
        self.store.execute("DELETE FROM rec_blocked_channels WHERE channel = ?", (name,))

    def blocked(self) -> set[str]:
        return {row[0] for row in self.store.query("SELECT channel FROM rec_blocked_channels")}
