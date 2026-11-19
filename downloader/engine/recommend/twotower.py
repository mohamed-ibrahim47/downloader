"""v7: the two-tower ranker, trained by hand on the owner's own lists (design v7, M and A2).

    h(title) = mean of the word vectors of the title's known words   (0 when none is known)
    item     v = tanh(Wi · h + bi)
    user     u = tanh(Wu · mean(h over your recent downloads) + bu)
    score    s = u · v

One training example is one list with a download: the downloaded video first, then every other video of that
list. The loss is softmax cross-entropy over the list, −log P(the downloaded one), plus an L2 penalty.

Backpropagation, from the scores back:
    dS_c = (p_c − [c is the download]) / B          softmax
    du = Σ_c dS_c · v_c        dv_c = dS_c · u       the dot product
    dz = d(out) · (1 − out²)                          tanh
    dW += dz ⊗ input   db += dz   d(input) = Wᵀ · dz  the dense layers

In the app the word vectors E are **frozen** (the owner's v4 vectors): a few hundred lists can't teach 5,000
words × 32 numbers, so only Wi, bi, Wu, bu learn. `train_e=True` unfreezes E; it exists so the lab's check case
(tests/engine/fixtures/two_tower_case.json, verified there against PyTorch) can check every gradient here too.

Pure Python on purpose: the exe has no numpy.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass

from . import evaluate
from .dataset import Row
from .profile import Profile
from .text import tokenize

HISTORY = 50  # newest downloads that make up "you"
K = 16  # tower output size: 2 × (16·32 + 16) = 1,056 weights with 32-number word vectors
LR = 0.05
MU = 0.9  # momentum, PyTorch's form: v = μ·v + g; w = w − lr·v
LAMBDA = 1e-3
EPOCHS = 40
BATCH = 8  # lists per step


@dataclass
class Example:
    hist: list[list[str]]  # token lists of the downloads before this list
    cands: list[list[str]]  # token lists of the list's videos, the downloaded one first


@dataclass
class TwoTower:
    wi: list[list[float]]  # [k][d]
    bi: list[float]
    wu: list[list[float]]
    bu: list[float]
    vectors_at: float | None = None  # which word vectors it was trained with (their "at")
    epochs: int = 0

    @property
    def k(self) -> int:
        return len(self.bi)

    def title_vec(self, tokens: list[str], E) -> list[float]:
        return title_vec(tokens, E)

    def item(self, tokens: list[str], E) -> list[float]:
        return _tower(self.wi, self.bi, title_vec(tokens, E))

    def user(self, hist: list[list[str]], E) -> list[float]:
        return _tower(self.wu, self.bu, _mean([title_vec(t, E) for t in hist[-HISTORY:]], E.dim))

    @staticmethod
    def score(u: list[float], v: list[float]) -> float:
        return sum(a * b for a, b in zip(u, v))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "TwoTower":
        return cls(**json.loads(text))


def title_vec(tokens: list[str], E) -> list[float]:
    """Plain mean of the known words' vectors; a repeated word counts each time."""
    known = [v for v in (E.get(t) for t in tokens) if v is not None]
    return _mean(known, E.dim)


def _mean(rows: list[list[float]], dim: int) -> list[float]:
    if not rows:
        return [0.0] * dim
    n = len(rows)
    return [sum(col) / n for col in zip(*rows)]


def _tower(W: list[list[float]], b: list[float], x: list[float]) -> list[float]:
    return [math.tanh(bi + sum(w * xj for w, xj in zip(row, x))) for row, bi in zip(W, b)]


def _known(tokens: list[str], E) -> list[str]:
    return [t for t in tokens if E.get(t) is not None]


def loss_and_grads(m: TwoTower, E, batch: list[Example], lam: float, train_e: bool = False,
                   cache: dict | None = None) -> tuple[float, dict]:
    """Mean loss over the batch + lam·(‖Wi‖² + ‖Wu‖² [+ Σ over the batch's words ‖E_w‖² when train_e]).

    `cache`: title vectors by token tuple, reused across steps (only valid while E is frozen)."""
    d, B = E.dim, len(batch)
    g = {"wi": [[0.0] * d for _ in m.wi], "bi": [0.0] * m.k, "wu": [[0.0] * d for _ in m.wu], "bu": [0.0] * m.k}
    gE: dict[str, list[float]] = {}

    def h_of(tokens):
        if cache is None:
            return title_vec(tokens, E)
        key = tuple(tokens)
        if key not in cache:
            cache[key] = title_vec(tokens, E)
        return cache[key]

    def to_words(tokens, dh):  # a title mean passes its gradient to each known word, 1/n each
        known = _known(tokens, E)
        for t in known:
            row = gE.setdefault(t, [0.0] * d)
            for j in range(d):
                row[j] += dh[j] / len(known)

    total, used = 0.0, set()
    for ex in batch:
        hist = ex.hist[-HISTORY:]
        hs = [h_of(t) for t in hist]
        ubar = _mean(hs, d)
        u = _tower(m.wu, m.bu, ubar)
        H = [h_of(c) for c in ex.cands]
        V = [_tower(m.wi, m.bi, h) for h in H]
        S = [TwoTower.score(u, v) for v in V]
        top = max(S)
        Z = sum(math.exp(s - top) for s in S)
        total += -(S[0] - top - math.log(Z))
        if train_e:
            for t in list(hist) + list(ex.cands):
                used.update(_known(t, E))
        du = [0.0] * m.k
        for c, (h, v, s) in enumerate(zip(H, V, S)):
            ds = (math.exp(s - top) / Z - (1.0 if c == 0 else 0.0)) / B
            for j in range(m.k):
                du[j] += ds * v[j]
            dz = [ds * u[j] * (1 - v[j] * v[j]) for j in range(m.k)]
            for i in range(m.k):
                g["bi"][i] += dz[i]
                row = g["wi"][i]
                for j in range(d):
                    row[j] += dz[i] * h[j]
            if train_e:
                to_words(ex.cands[c], [sum(m.wi[i][j] * dz[i] for i in range(m.k)) for j in range(d)])
        dz = [du[j] * (1 - u[j] * u[j]) for j in range(m.k)]
        for i in range(m.k):
            g["bu"][i] += dz[i]
            row = g["wu"][i]
            for j in range(d):
                row[j] += dz[i] * ubar[j]
        if train_e and hist:
            dubar = [sum(m.wu[i][j] * dz[i] for i in range(m.k)) / len(hist) for j in range(d)]
            for t in hist:
                to_words(t, dubar)

    reg = sum(w * w for row in m.wi for w in row) + sum(w * w for row in m.wu for w in row)
    for name in ("wi", "wu"):
        for grow, wrow in zip(g[name], getattr(m, name)):
            for j in range(d):
                grow[j] += 2 * lam * wrow[j]
    if train_e:
        for t in used:
            e = E.get(t)
            reg += sum(x * x for x in e)
            row = gE.setdefault(t, [0.0] * d)
            for j in range(d):
                row[j] += 2 * lam * e[j]
        g["E"] = gE
    return total / B + lam * reg, g


def step(m: TwoTower, vel: dict, grads: dict, lr: float, mu: float, E=None) -> None:
    """SGD with momentum, in place, PyTorch's form (v = μ·v + g; w = w − lr·v). With `E`, word rows move too,
    including rows with no gradient this step: their velocity still carries them, as in PyTorch."""
    for name in ("wi", "bi", "wu", "bu"):
        w, gr = getattr(m, name), grads[name]
        v = vel.setdefault(name, [[0.0] * len(r) for r in gr] if isinstance(gr[0], list) else [0.0] * len(gr))
        rows = zip(w, gr, v) if isinstance(gr[0], list) else [(w, gr, v)]
        for wr, gr_, vr in rows:
            for j in range(len(wr)):
                vr[j] = mu * vr[j] + gr_[j]
                wr[j] -= lr * vr[j]
    if E is not None and "E" in grads:
        vE = vel.setdefault("E", {})
        for t in set(vE) | set(grads["E"]):
            gr_ = grads["E"].get(t)
            vr = vE.setdefault(t, [0.0] * E.dim)
            row = E.get(t)
            for j in range(E.dim):
                vr[j] = mu * vr[j] + (gr_[j] if gr_ else 0.0)
                row[j] -= lr * vr[j]


def init(d: int, k: int, rng: random.Random) -> TwoTower:
    return TwoTower([[rng.gauss(0, 0.1) for _ in range(d)] for _ in range(k)], [0.0] * k,
                    [[rng.gauss(0, 0.1) for _ in range(d)] for _ in range(k)], [0.0] * k)


def fit(examples: list[Example], E, k: int = K, epochs: int = EPOCHS, seed: int = 0, lr: float = LR,
        mu: float = MU, lam: float = LAMBDA, batch: int = BATCH) -> TwoTower:
    """Train Wi, bi, Wu, bu with E frozen. Seeded, so the same log gives the same model."""
    rng = random.Random(seed)
    m = init(E.dim, k, rng)
    vel: dict = {}
    cache: dict = {}
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            _, g = loss_and_grads(m, E, [examples[i] for i in order[s:s + batch]], lam, cache=cache)
            step(m, vel, g, lr, mu)
    m.epochs = epochs
    return m


# ---- training examples from the log (design A3) ----------------------------------------------

def titled_lists(rows: list[Row]) -> dict[float, list[Row]]:
    """Refresh -> its rows, only for lists where every row has a title (v7 can't read the others)."""
    lists: dict[float, list[Row]] = {}
    for r in rows:
        lists.setdefault(r.refresh, []).append(r)
    return {ref: rs for ref, rs in lists.items() if all(r.title for r in rs)}


def history_of(profile: Profile) -> list[tuple[float, list[str]]]:
    """(when, tokens) of every download in the profile (weight > 0; dismissals have negative weight)."""
    return [(d.added_at, d.tokens) for d in profile.docs if d.weight > 0]


def examples(rows: list[Row], history: list[tuple[float, list[str]]]) -> list[Example]:
    """One example per download in a titled list: that video first, then the list's videos that were dismissed
    or scrolled past (another download in the same list is never a "no"). "You" at that moment = the newest
    HISTORY downloads finished before the list was made."""
    hist_sorted = sorted(history, key=lambda h: h[0])
    out = []
    for refresh, rs in titled_lists(rows).items():
        hist = [tok for at, tok in hist_sorted if at < refresh][-HISTORY:]
        negatives = [tokenize(r.title) for r in rs if r.label == 0]
        for r in rs:
            if r.label == 1:
                out.append(Example(hist=hist, cands=[tokenize(r.title)] + negatives))
    return out


# ---- the contest: does v7 beat the ranker on the page? (design A4, A7) -------------------------

@dataclass
class Contest:
    lists: int  # titled test lists with a download
    v7: float  # mean NDCG@10 on them
    rival: float
    wins: bool
    reason: str  # in the owner's words
    model: TwoTower | None = None  # the v7 trained on the older lists (v8's contest re-scores it, design v8 A4)


def ndcg(m: TwoTower, E, lists: dict[float, list[Row]], history) -> float:
    hist_sorted = sorted(history, key=lambda h: h[0])
    total = 0.0
    for refresh, rs in lists.items():
        u = m.user([tok for at, tok in hist_sorted if at < refresh][-HISTORY:], E)
        scored = sorted(rs, key=lambda r: (-m.score(u, m.item(tokenize(r.title), E)), r.video_id))
        total += evaluate.ndcg_at([r.label for r in scored])
    return total / len(lists) if lists else 0.0


def contest(rows: list[Row], history, E, rival: str, models: dict) -> Contest:
    """Train v7 on the older lists, then judge it and `rival` (the ranker `evaluate.choose` picked) on the
    newest lists in which every video has a title. v7 must win outright: on a tie the rival stays."""
    downloads = sum(r.label for r in rows)
    if len(rows) < evaluate.MIN_ROWS or downloads < evaluate.MIN_DOWNLOADS:
        return Contest(0, 0.0, 0.0, False, f"Learning: {len(rows)} of {evaluate.MIN_ROWS} rows, "
                                           f"{downloads} of {evaluate.MIN_DOWNLOADS} downloads")
    train, test = evaluate.split(rows)
    judged = {ref: rs for ref, rs in titled_lists(test).items() if any(r.label for r in rs)}
    if len(judged) < evaluate.MIN_TEST_LISTS:
        return Contest(len(judged), 0.0, 0.0, False,
                       f"Waiting for lists to test on: {len(judged)} of {evaluate.MIN_TEST_LISTS} recent lists "
                       "with titles have a download")
    ex = examples(train, history)
    if not ex:
        return Contest(len(judged), 0.0, 0.0, False, "Nothing to learn from yet: no older list with titles has a download")
    m = fit(ex, E)
    v7 = ndcg(m, E, judged, history)
    rv = evaluate.rank_lists([r for rs in judged.values() for r in rs], models)[1].get(rival, 0.0)
    if v7 > rv:
        return Contest(len(judged), v7, rv, True, f"v7 beats {rival} on your newest lists (NDCG {v7:.2f} vs {rv:.2f})",
                       model=m)
    return Contest(len(judged), v7, rv, False, f"v7 didn't beat {rival} yet (NDCG {v7:.2f} vs {rv:.2f})", model=m)
