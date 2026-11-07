"""The recommender in one object: profile → candidates → ranking, plus the feedback (spec B1, B7).

The page talks only to this. `refresh` does the network work and runs in the pool; everything else reads
the local database and runs on the GUI thread.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

from .. import youtube
from ..store import Store
from .candidates import Candidate, fetch, keep, queries
from .log import RecLog
from .profile import Profile, build_profile
from .rank import TOP_K, Suggestion, build_model, profile_words, rank

REFRESH_AFTER = 6 * 3600  # seconds: opening the page searches again only when the cache is older


@dataclass
class View:
    """One render's worth: built from a single read of the downloads."""
    ready: bool
    suggestions: list[Suggestion] = field(default_factory=list)
    words: list[tuple[str, float]] = field(default_factory=list)


class Recommender:
    def __init__(self, store: Store, search=None, clock=time.time, rng: random.Random | None = None):
        self.store = store
        self.log = RecLog(store)
        # Looked up now, not when Python read this file, so a stubbed youtube.search is the one used.
        self._search = search if search is not None else youtube.search
        self._clock = clock
        self._rng = rng or random.Random()
        self._refreshing = threading.Lock()
        # Per refresh (GUI thread only): videos already logged as shown, and the discovery pick.
        self._generation: float | None = None
        self._logged: set[str] = set()
        self._discovery: str | None = None

    def profile(self) -> Profile:
        return build_profile(self.store.all(), self.log.dismissed(), self._clock())

    def ready(self) -> bool:
        return self.profile().ready

    def needs_refresh(self) -> bool:
        last = self.log.last_refresh()
        return last is None or self._clock() - last > REFRESH_AFTER

    def refresh(self) -> None:
        """Search YouTube and replace the cache. Blocking.

        If every search fails, or none finds anything, the old cache stays. If a refresh is already
        running (the page was opened twice), this one returns at once without searching.
        """
        if not self._refreshing.acquire(blocking=False):
            return
        try:
            profile = self.profile()
            if not profile.ready:
                return  # nothing would be shown, so don't search
            cands = fetch(queries(profile, self.log.blocked()), self._search)  # raises when offline
            if cands:
                self.log.save_candidates(cands, self._clock())
            else:
                self.log.mark_refreshed(self._clock())
        finally:
            self._refreshing.release()

    def view(self, k: int = TOP_K, words: int = 0, log: bool = True) -> View:
        """Suggestions (and, if asked, the profile's top `words`) from one read of the downloads."""
        now = self._clock()
        downloads = self.store.all()
        dismissed = self.log.dismissed()
        profile = build_profile(downloads, dismissed, now)
        if not profile.ready:
            return View(False)
        have = {d.video_id for d in downloads if d.video_id}
        self.log.mark_downloaded(have, now)
        cands = keep(self.log.candidates(), have, {x.video_id for x in dismissed}, self.log.blocked())
        self._start_generation()
        model = build_model(profile, cands)
        picks = rank(profile, cands, k, self._rng, discovery_id=self._discovery, model=model)
        self._discovery = next((s.candidate.video_id for s in picks if s.discovery), self._discovery)
        if log:
            self._log_shown(picks, profile)
        return View(True, picks, profile_words(profile, cands, words, model=model) if words else [])

    def _start_generation(self) -> None:
        """A new refresh means a new list: forget what was logged and the old discovery pick."""
        last = self.log.last_refresh()
        if last != self._generation:
            self._generation, self._logged, self._discovery = last, set(), None

    def _log_shown(self, picks: list[Suggestion], profile: Profile) -> None:
        """One `shown` row per video per refresh: a redraw after a Dismiss is not a new impression."""
        new = [(i, s) for i, s in enumerate(picks, start=1) if s.candidate.video_id not in self._logged]
        if new:
            self.log.record_many([("shown", s.candidate, s.score, i, _features(s, profile, i)) for i, s in new],
                                 self._clock())
            self._logged.update(s.candidate.video_id for _, s in new)

    def suggestions(self, k: int = TOP_K) -> list[Suggestion]:
        return self.view(k).suggestions

    def words(self, n: int = 15) -> list[tuple[str, float]]:
        return self.view(words=n, log=False).words

    def dismiss(self, s: Suggestion) -> None:
        self.log.record("dismissed", s.candidate, self._clock(), s.score)

    def block_channel(self, name: str) -> None:
        self.log.block_channel(name, self._clock())

    def unblock_channel(self, name: str) -> None:
        self.log.unblock_channel(name)

    def blocked_channels(self) -> list[str]:
        return sorted(self.log.blocked())


def _features(s: Suggestion, profile: Profile, rank_: int) -> dict:
    """One training row for v2 (a learned ranker): the inputs v1 used, and where it put the video."""
    return {"similarity": s.similarity, "boost": s.boost,
            "affinity": profile.channels.get(s.candidate.channel, 0.0) if s.candidate.channel else 0.0,
            "duration": s.candidate.duration, "rank": rank_, "discovery": s.discovery, "shared": len(s.shared)}


__all__ = ["REFRESH_AFTER", "Recommender", "View", "Candidate"]
