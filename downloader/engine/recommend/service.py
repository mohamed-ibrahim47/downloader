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
from ..settings import Settings
from ..store import Store
from . import evaluate, starter
from .candidates import Candidate, fetch, keep, plan
from .dataset import rows as training_rows
from .features import NAMES, from_suggestion
from .learn import Weights, available_trainers, trainer
from .neural import Net, NetTrainer
from .log import RecLog
from .profile import Profile, build_profile
from .rank import TOP_K, Suggestion, build_model, profile_words, rank

REFRESH_AFTER = 6 * 3600  # seconds: opening the page searches again only when the cache is older
MODEL_KEY = "rec_model"  # settings key: the latest training, its evaluation, and which ranker is on


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
        self._training = threading.Lock()
        # Per refresh (GUI thread only): videos already logged as shown, and the discovery pick.
        self._generation: float | None = None
        self._logged: set[str] = set()
        self._discovery: str | None = None

    def profile(self) -> Profile:
        return build_profile(self.store.all(), self.log.dismissed(), self._clock())

    def ready(self) -> bool:
        return self.profile().ready

    def cache_age(self) -> float | None:
        """Seconds since the last refresh, on this recommender's clock; None if it never ran."""
        last = self.log.last_refresh()
        return None if last is None else max(0.0, self._clock() - last)

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
            cands = fetch(plan(profile, self.log.blocked()), self._search)  # raises when offline
            if cands:
                self.log.save_candidates(cands, self._clock())
            else:
                self.log.mark_refreshed(self._clock())
            self.train()  # the list the owner just saw is settled now: learn from it
        finally:
            self._refreshing.release()

    # ---- the learned rankers (v2, v3, the starter) -------------------------------------

    def train(self) -> None:
        """Train v2 and v3 on the log, measure them against v1, and save which one should rank. Blocking.

        Runs after each refresh (in the pool) and when the owner picks another trainer. Also trains every
        installed v2 trainer on all rows, so the page can show their weights side by side. v3's final net is
        trained on all rows only when it won, for the number of passes its held-back run found.
        """
        with self._training:
            settings = Settings.load(self.store)
            rows = training_rows(self.log.events(), open_refresh=self.log.last_refresh())
            chosen = trainer(settings.rec_trainer)
            try:
                rep = evaluate.evaluate(rows, chosen)
                ranker, why = evaluate.choose(rep)
                compare, net = {}, None
                if len({r.label for r in rows}) == 2:  # a yes and a no: something to learn
                    X, y, w = [r.x for r in rows], [r.label for r in rows], [r.weight for r in rows]
                    compare = {name: trainer(name).fit(NAMES, X, y, w).to_json() for name in available_trainers()}
                    if ranker == "v3" and rep.net is not None:
                        net = NetTrainer(epochs=rep.net.epochs).fit(NAMES, X, y, w).to_json()
            except Exception as e:  # a training bug must never take the page down: v1 keeps ranking
                rep, ranker, why, compare, net = evaluate.Report(len(rows), 0), "v1", f"Training failed: {e}", {}, None
            final = compare.get(chosen.name)
            if (ranker == "v2" and final is None) or (ranker == "v3" and net is None):
                ranker = "v1"
            self.store.set_setting(MODEL_KEY, {
                "active": ranker != "v1", "ranker": ranker, "reason": why, "weights": final, "net": net,
                "compare": compare,
                "report": {"rows": rep.rows, "downloads": rep.downloads, "test_lists": rep.test_lists,
                           "ndcg": rep.ndcg, "hit3": rep.hit3, "auc": rep.auc, "log_loss": rep.log_loss,
                           "v3_auc": rep.v3_auc, "v3_log_loss": rep.v3_log_loss,
                           "v3_epochs": rep.net.epochs if rep.net else 0},
                "trained_at": self._clock()})

    def model(self) -> dict | None:
        """The latest training, as saved: for the "Why this order?" view."""
        return self.store.get_setting(MODEL_KEY)

    def _learned(self) -> tuple[str, Weights | Net] | None:
        """The ranker that ranks the page and its model, or None when v1 does (design v3, D4).

        The owner's own trained model when it passed the gate; else the starter while their log is still too
        small to judge; else v1. The setting off means v1, always.
        """
        if not Settings.load(self.store).rec_learned:
            return None
        saved = self.model() or {}
        if saved.get("active"):
            if saved.get("ranker") == "v3" and saved.get("net"):
                return "v3", Net.from_json(saved["net"])
            return "v2", Weights.from_json(saved["weights"])  # saved by 1.3.0 (no "ranker") is v2 too
        report = saved.get("report") or {}
        small = report.get("rows", 0) < evaluate.MIN_ROWS or report.get("downloads", 0) < evaluate.MIN_DOWNLOADS
        if small and starter.RANKER:
            return f"starter-{starter.RANKER}", starter.model()
        return None

    def ranker(self) -> str:
        """"v1", "v2", "v3", "starter-v2" or "starter-v3"."""
        learned = self._learned()
        return learned[0] if learned else "v1"

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
        learned = self._learned()
        name, m = learned if learned else ("v1", None)
        scorer = (lambda s, affinity: m.prob(from_suggestion(s, affinity))) if m is not None else None
        picks = rank(profile, cands, k, self._rng, discovery_id=self._discovery, model=model, scorer=scorer)
        self._discovery = next((s.candidate.video_id for s in picks if s.discovery), self._discovery)
        if log:
            self._log_shown(picks, profile, name)
        return View(True, picks, profile_words(profile, cands, words, model=model) if words else [])

    def _start_generation(self) -> None:
        """A new refresh means a new list: forget what was logged and the old discovery pick."""
        last = self.log.last_refresh()
        if last != self._generation:
            self._generation, self._logged, self._discovery = last, set(), None

    def _log_shown(self, picks: list[Suggestion], profile: Profile, ranker: str) -> None:
        """One `shown` row per video per refresh: a redraw after a Dismiss is not a new impression."""
        new = [(i, s) for i, s in enumerate(picks, start=1) if s.candidate.video_id not in self._logged]
        if new:
            self.log.record_many([("shown", s.candidate, s.score, i, _features(s, profile, i, ranker)) for i, s in new],
                                 self._clock(), refresh=self._generation)
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


def _features(s: Suggestion, profile: Profile, rank_: int, ranker: str = "v1") -> dict:
    """One training row for v2 (a learned ranker): the inputs v1 used, and where it put the video."""
    return {"similarity": s.similarity, "boost": s.boost,
            "affinity": profile.channels.get(s.candidate.channel, 0.0) if s.candidate.channel else 0.0,
            "duration": s.candidate.duration, "rank": rank_, "discovery": s.discovery, "shared": len(s.shared),
            "kind": s.candidate.kind, "ranker": ranker}


__all__ = ["REFRESH_AFTER", "Recommender", "View", "Candidate"]
