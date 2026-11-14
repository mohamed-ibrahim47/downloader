"""The recommender in one object: profile → candidates → ranking, plus the feedback (spec B1, B7).

The page talks only to this. `refresh` does the network work and runs in the pool; everything else reads
the local database and runs on the GUI thread.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field

from .. import youtube
from ..settings import Settings
from ..store import Store
from . import embed, evaluate, starter, starter_vectors
from .candidates import Candidate, fetch, keep, plan
from .dataset import rows as training_rows
from .features import NAMES, from_suggestion
from .learn import Weights, available_trainers, trainer
from .neural import Net, NetTrainer
from .log import RecLog
from .meaning import MeaningIndex, v4_score
from .profile import Dismissed, Profile, build_profile
from .rank import TOP_K, Suggestion, build_model, profile_words, rank
from .text import tokenize
from .tfidf import fit_idf

REFRESH_AFTER = 6 * 3600  # seconds: opening the page searches again only when the cache is older
MODEL_KEY = "rec_model"  # settings key: the latest training, its evaluation, and which ranker is on
VECTORS_KEY = "rec_vectors_meta"  # settings key: when the owner's own vectors were trained, on how many titles
BANNER_KEY = "rec_meaning_banner"  # settings key: "done" once the owner closed the "match by meaning" banner
RETRAIN_EVERY = 300  # new titles seen before the vectors are trained again on the owner's own titles


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
        self._retraining = threading.Lock()
        self._vectors: embed.Vectors | None = None
        self._vectors_loaded = False
        # Per refresh (GUI thread only): videos already logged as shown, and the discovery pick.
        self._generation: float | None = None
        self._logged: set[str] = set()
        self._discovery: str | None = None

    def profile(self) -> Profile:
        return build_profile(self.store.all(), self.log.dismissed(), self._clock(),
                             files=Settings.load(self.store).rec_files)

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
            events = self.log.events()
            vectors = self.vectors() if settings.rec_meaning else None
            meaning_fn = self._meaning_at(events, vectors, settings) if vectors else (lambda _t, _r: 0.0)
            rows = training_rows(events, open_refresh=self.log.last_refresh(), meaning=meaning_fn)
            chosen = trainer(settings.rec_trainer)
            meaning_on = vectors is not None
            try:
                rep = evaluate.evaluate(rows, chosen, mix=starter_vectors.MEANING_MIX if meaning_on else None)
                ranker, why = evaluate.choose(rep, meaning_on, starter_vectors.DEFAULT_ON)
                first = evaluate.base(rep, meaning_on, starter_vectors.DEFAULT_ON)[0]
                compare, net = {}, None
                if len({r.label for r in rows}) == 2:  # a yes and a no: something to learn
                    X, y, w = [r.x for r in rows], [r.label for r in rows], [r.weight for r in rows]
                    compare = {name: trainer(name).fit(NAMES, X, y, w).to_json() for name in available_trainers()}
                    if ranker == "v3" and rep.net is not None:
                        net = NetTrainer(epochs=rep.net.epochs).fit(NAMES, X, y, w).to_json()
            except Exception as e:  # a training bug must never take the page down: v1 keeps ranking
                rep, ranker, why, compare, net = evaluate.Report(len(rows), 0), "v1", f"Training failed: {e}", {}, None
                first = "v1"
            final = compare.get(chosen.name)
            if (ranker == "v2" and final is None) or (ranker == "v3" and net is None):
                ranker = first
            self.store.set_setting(MODEL_KEY, {
                "active": ranker in ("v2", "v3"), "ranker": ranker, "base": first, "reason": why, "weights": final,
                "net": net,
                "compare": compare,
                "report": {"rows": rep.rows, "downloads": rep.downloads, "test_lists": rep.test_lists,
                           "ndcg": rep.ndcg, "hit3": rep.hit3, "auc": rep.auc, "log_loss": rep.log_loss,
                           "v3_auc": rep.v3_auc, "v3_log_loss": rep.v3_log_loss,
                           "v3_epochs": rep.net.epochs if rep.net else 0},
                "trained_at": self._clock()})
        self.retrain_vectors()  # on its own clock: only after RETRAIN_EVERY new titles

    # ---- the word vectors (v4) ---------------------------------------------------------

    def vectors(self) -> embed.Vectors | None:
        """The owner's own vectors once trained, else the starter's, else None."""
        if not self._vectors_loaded:
            self._vectors = self.log.load_vectors() or starter_vectors.vectors()
            self._vectors_loaded = True
        return self._vectors

    def vectors_meta(self) -> dict:
        """For "Why this order?": where the vectors came from, how many words, how many of your titles."""
        meta = self.store.get_setting(VECTORS_KEY)
        if meta:
            return meta
        v = self.vectors()
        return {"source": "starter", "words": len(v) if v else 0, "titles": 0, "at": None}

    def _user_titles(self) -> list[str]:
        """Every title the owner's app has seen: suggestions shown, downloads, and file names that join."""
        settings = Settings.load(self.store)
        titles = set(self.log.titles())
        profile = build_profile(self.store.all(), [], self._clock(), files=settings.rec_files)
        titles.update(d.title for d in profile.docs if d.title)
        return sorted(titles)

    def retrain_vectors(self) -> bool:
        """Train the vectors again on the owner's own titles, starting from the current ones. Blocking.

        Runs only when RETRAIN_EVERY new titles have been seen since the last time, never twice at once, and
        a failure keeps the old vectors. True when it trained.
        """
        if not self._retraining.acquire(blocking=False):
            return False
        try:
            titles = self._user_titles()
            done = (self.store.get_setting(VECTORS_KEY) or {}).get("titles", 0)
            if len(titles) - done < RETRAIN_EVERY:
                return False
            try:
                v = embed.train([tokenize(t) for t in titles], start=self.vectors(), epochs=embed.RETRAIN_EPOCHS,
                                min_count=embed.USER_MIN_COUNT)
            except Exception:  # a training bug must never take the page down: the old vectors stay
                return False
            if not len(v):
                return False
            self.log.save_vectors(v)
            self.store.set_setting(VECTORS_KEY, {"source": "own", "titles": len(titles), "words": len(v),
                                                 "at": self._clock()})
            self._vectors, self._vectors_loaded = v, True
            return True
        finally:
            self._retraining.release()

    def _meaning_at(self, events: list[tuple], vectors: embed.Vectors, settings: Settings):
        """meaning(title, refresh) for training rows: the profile as it was at that refresh, today's vectors.

        Rows of one refresh share one profile, so it is built once per refresh. Downloads count from when they
        were added, dismissals from when they were logged.
        """
        downloads = self.store.all()
        dismissals = [(e[4], Dismissed(e[1], e[6] or "", None)) for e in events if e[2] == "dismissed"]
        shown: dict[float, list[str]] = {}
        for e in events:
            if e[2] == "shown" and e[5] is not None:
                shown.setdefault(e[5], []).append(e[6] or "")
        cache: dict[float, MeaningIndex] = {}

        def meaning(title: str, refresh: float) -> float:
            if refresh not in cache:
                then = [d for d in downloads if d.added_at <= refresh]
                gone = [x for at, x in dismissals if at <= refresh]
                profile = build_profile(then, gone, refresh, files=settings.rec_files)
                idf = fit_idf([d.tokens for d in profile.docs] + [tokenize(t) for t in shown.get(refresh, [])])
                cache[refresh] = MeaningIndex(profile, vectors, idf)
            return cache[refresh].score(tokenize(title))
        return meaning

    # ---- the banner ------------------------------------------------------------------

    def uses_meaning(self) -> bool:
        """Is the page ranked with meaning: v4, or a learned ranker that has meaning as an input?"""
        if not Settings.load(self.store).rec_meaning or self.vectors() is None:
            return False
        return self.ranker() in ("v4", "v2", "v3")

    def banner_due(self) -> bool:
        return self.uses_meaning() and self.store.get_setting(BANNER_KEY) != "done"

    def banner_done(self) -> None:
        self.store.set_setting(BANNER_KEY, "done")

    def example(self) -> tuple[str, str] | None:
        return starter_vectors.EXAMPLE

    def nearest_words(self, n_words: int = 5, n: int = 4) -> list[tuple[str, list[str]]]:
        """The profile's top words that have vectors, each with its `n` nearest words."""
        v = self.vectors()
        if v is None:
            return []
        out = []
        for word, _score in self.words(40):
            if v.get(word) is not None:
                out.append((word, v.nearest(word, n)))
                if len(out) == n_words:
                    break
        return out

    def model(self) -> dict | None:
        """The latest training, as saved: for the "Why this order?" view."""
        return self.store.get_setting(MODEL_KEY)

    def _learned(self) -> tuple[str, Weights | Net | None]:
        """The ranker that ranks the page and its model (design v3 D4, v4 E7). v1 and v4 have no model.

        The owner's own trained model when it passed the gate; else the starter while their log is still too
        small to judge; else the base, v4 or v1. "Use the learned ranker" off leaves only the base: v4 learns
        no weights from the owner's clicks. "Match by meaning" off (or no vectors) makes the base v1.
        """
        settings = Settings.load(self.store)
        saved = self.model() or {}
        meaning = settings.rec_meaning and self.vectors() is not None
        first = saved.get("base") or ("v4" if starter_vectors.DEFAULT_ON else "v1")
        if not meaning:
            first = "v1"
        if not settings.rec_learned:
            return first, None
        if saved.get("active") and not _current_inputs(saved):
            return first, None  # saved by an older version with other inputs: the base until the next training
        if saved.get("active"):
            if saved.get("ranker") == "v3" and saved.get("net"):
                return "v3", Net.from_json(saved["net"])
            return "v2", Weights.from_json(saved["weights"])  # saved by 1.3.0 (no "ranker") is v2 too
        report = saved.get("report") or {}
        small = report.get("rows", 0) < evaluate.MIN_ROWS or report.get("downloads", 0) < evaluate.MIN_DOWNLOADS
        if small and starter.RANKER:
            return f"starter-{starter.RANKER}", starter.model()
        return first, None

    def ranker(self) -> str:
        """"v1", "v4", "v2", "v3", "starter-v2" or "starter-v3"."""
        return self._learned()[0]

    def view(self, k: int = TOP_K, words: int = 0, log: bool = True) -> View:
        """Suggestions (and, if asked, the profile's top `words`) from one read of the downloads."""
        now = self._clock()
        downloads = self.store.all()
        dismissed = self.log.dismissed()
        profile = build_profile(downloads, dismissed, now, files=Settings.load(self.store).rec_files)
        if not profile.ready:
            return View(False)
        have = {d.video_id for d in downloads if d.video_id}
        self.log.mark_downloaded(have, now)
        cands = keep(self.log.candidates(), have, {x.video_id for x in dismissed}, self.log.blocked())
        self._start_generation()
        model = build_model(profile, cands)
        name, m = self._learned()
        vectors = self.vectors() if Settings.load(self.store).rec_meaning else None
        index = MeaningIndex(profile, vectors, model.idf) if vectors is not None else None
        if m is not None:
            scorer = lambda s, affinity: m.prob(from_suggestion(s, affinity))  # noqa: E731
        elif name == "v4":
            scorer = lambda s, affinity: v4_score(s.similarity, s.meaning, affinity,  # noqa: E731
                                                  starter_vectors.MEANING_MIX)
        else:
            scorer = None
        picks = rank(profile, cands, k, self._rng, discovery_id=self._discovery, model=model, scorer=scorer,
                     meaning=index)
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


def _current_inputs(saved: dict) -> bool:
    """Was this saved model trained on today's inputs (`features.NAMES`)?"""
    text = saved.get("net") if saved.get("ranker") == "v3" and saved.get("net") else saved.get("weights")
    try:
        return json.loads(text).get("names") == NAMES
    except (TypeError, ValueError, AttributeError):
        return False


def _features(s: Suggestion, profile: Profile, rank_: int, ranker: str = "v1") -> dict:
    """One training row for v2 (a learned ranker): the inputs v1 used, and where it put the video."""
    return {"similarity": s.similarity, "boost": s.boost,
            "affinity": profile.channels.get(s.candidate.channel, 0.0) if s.candidate.channel else 0.0,
            "duration": s.candidate.duration, "rank": rank_, "discovery": s.discovery, "shared": len(s.shared),
            "kind": s.candidate.kind, "ranker": ranker,
            "meaning": s.meaning}


__all__ = ["REFRESH_AFTER", "Recommender", "View", "Candidate"]
