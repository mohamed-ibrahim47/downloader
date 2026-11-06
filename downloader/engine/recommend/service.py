"""The recommender in one object: profile → candidates → ranking, plus the feedback (spec B1, B7).

The page talks only to this. `refresh` does the network work and runs in the pool; everything else reads
the local database and is quick.
"""
from __future__ import annotations

import random
import time

from .. import youtube
from ..store import Store
from .candidates import Candidate, fetch, keep, queries
from .log import RecLog
from .profile import Profile, build_profile
from .rank import TOP_K, Suggestion, profile_words, rank

REFRESH_AFTER = 6 * 3600  # seconds: opening the page searches again only when the cache is older


class Recommender:
    def __init__(self, store: Store, search=None, clock=time.time, rng: random.Random | None = None):
        self.store = store
        self.log = RecLog(store)
        # Looked up now, not when Python read this file, so a stubbed youtube.search is the one used.
        self._search = search if search is not None else youtube.search
        self._clock = clock
        self._rng = rng or random.Random()

    def profile(self) -> Profile:
        return build_profile(self.store.all(), self.log.dismissed(), self._clock())

    def needs_refresh(self) -> bool:
        last = self.log.last_refresh()
        return last is None or self._clock() - last > REFRESH_AFTER

    def refresh(self) -> None:
        """Search YouTube and replace the cache. Blocking. If every search fails, the old cache stays."""
        profile = self.profile()
        if not profile.ready:
            return  # nothing would be shown, so don't search
        cands = fetch(queries(profile), self._search)  # raises DownloadError when offline
        self.log.save_candidates(cands, self._clock())

    def _state(self) -> tuple[Profile, list[Candidate]]:
        """The profile and the cached candidates the owner hasn't seen through yet."""
        now = self._clock()
        downloads = self.store.all()
        dismissed = self.log.dismissed()
        profile = build_profile(downloads, dismissed, now)
        have = {d.video_id for d in downloads if d.video_id}
        self.log.mark_downloaded(have, now)
        cands = keep(self.log.candidates(), have, {x.video_id for x in dismissed}, self.log.blocked())
        return profile, cands

    def suggestions(self, k: int = TOP_K) -> list[Suggestion]:
        profile, cands = self._state()
        if not profile.ready:
            return []
        picks = rank(profile, cands, k, self._rng)
        self.log.record_many([("shown", s.candidate, s.score, i, _features(s, profile, i))
                              for i, s in enumerate(picks, start=1)], self._clock())
        return picks

    def dismiss(self, s: Suggestion) -> None:
        self.log.record("dismissed", s.candidate, self._clock(), s.score)

    def block_channel(self, name: str) -> None:
        self.log.block_channel(name, self._clock())

    def unblock_channel(self, name: str) -> None:
        self.log.unblock_channel(name)

    def blocked_channels(self) -> list[str]:
        return sorted(self.log.blocked())

    def words(self, n: int = 15) -> list[tuple[str, float]]:
        profile, cands = self._state()
        return profile_words(profile, cands, n)


def _features(s: Suggestion, profile: Profile, rank_: int) -> dict:
    """One training row for v2 (a learned ranker): the inputs v1 used, and where it put the video."""
    return {"similarity": s.similarity, "boost": s.boost,
            "affinity": profile.channels.get(s.candidate.channel, 0.0) if s.candidate.channel else 0.0,
            "duration": s.candidate.duration, "rank": rank_, "discovery": s.discovery, "shared": len(s.shared)}
