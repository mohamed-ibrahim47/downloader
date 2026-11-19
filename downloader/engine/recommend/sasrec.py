"""v8: the sequence ranker (SASRec over titles), trained by hand on the owner's own lists (design v8, M and A2).

v7 averages your downloads, so their order is lost. v8 reads them as a sequence, oldest first:

    item     x = tanh(Wi · h(title) + bi)                     v7's item tower (h = mean of the word vectors)
    input    z_t = x_t + p[L − n + t]                         a learned vector per position, counted from the newest
    attend   q, k, v = z·Wq, z·Wk, z·Wv
             a_t = softmax over s ≤ t of (q_t · k_s / √k)    step t looks back at the steps it finds relevant
    block    s_t = LN(z_t + Σ a_ts v_s)                        LN: (x − mean)/sqrt(var + 1e-8), then ·g + be
             f_t = LN(s_t + relu(s_t·W1 + b1)·W2 + b2)
    user     u = f at the newest step (no download yet: u = 0, every score 0)
    score    u · x(candidate)

Loss per list with a download: −log P(the downloaded video) over the list (as v7), plus α × SASRec's own task,
"from step t, pick download t+1 out of it and a few videos you were shown but didn't download", averaged over
the steps. Plus an L2 penalty on the matrices and the position table.

Backpropagation goes the same road backwards: softmax → scores → f → LN₂ → feed-forward → LN₁ → attention
(each row's softmax: dlogit = a·(da − Σ a·da)) → q, k, v → z → positions and items → tanh → Wi.

The word vectors E are frozen in the app (the owner's v4 vectors), as for v7. `train_e=True` unfreezes them so
the lab's check case (tests/engine/fixtures/sasrec_case.json, verified there against PyTorch) can check every
gradient here too. Pure Python on purpose: the exe has no numpy.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field

from .dataset import Row
from .text import tokenize
from .twotower import title_vec, titled_lists

HISTORY = 20  # L: the newest downloads read as a sequence (attention costs grow with L², and this is plain Python)
K = 16  # model width
NEG_SEQ = 4  # wrong answers per step in the next-download task
ALPHA = 1.0  # weight of the next-download task (the lab's choice, see the walkthrough)
LR = 0.05
MU = 0.9  # momentum, PyTorch's form: v = μ·v + g; w = w − lr·v
LAMBDA = 1e-3
EPOCHS = 40  # at most; a big log gets fewer (epochs_for)
MIN_EPOCHS = 5
WORK = 1500  # example passes per fit: about a minute at HISTORY 20 with the next-download task on
BATCH = 8  # lists per step
EPS = 1e-8

MATRICES = ("wi", "p", "wq", "wk", "wv", "w1", "w2")  # L2-penalised
VECTORS = ("bi", "b1", "b2", "g1", "be1", "g2", "be2")
FIELDS = MATRICES + VECTORS


@dataclass
class Example:
    hist: list[list[str]]  # token lists of the downloads before this list, oldest first
    cands: list[list[str]]  # token lists of the list's videos, the downloaded one first
    seq_negs: list[list[list[str]]] = field(default_factory=list)  # [t]: wrong answers next to download t+1


@dataclass
class SasRec:
    wi: list[list[float]]  # [k][d]
    bi: list[float]
    p: list[list[float]]  # [L][k]; row L−1 = the newest download
    wq: list[list[float]]  # [k][k], used as z·Wq
    wk: list[list[float]]
    wv: list[list[float]]
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]
    g1: list[float]
    be1: list[float]
    g2: list[float]
    be2: list[float]
    vectors_at: float | None = None  # which word vectors it was trained with (their "at")
    epochs: int = 0

    @property
    def k(self) -> int:
        return len(self.bi)

    @property
    def L(self) -> int:
        return len(self.p)

    def item(self, tokens: list[str], E) -> list[float]:
        return _tower(self.wi, self.bi, title_vec(tokens, E))

    def user(self, hist: list[list[str]], E) -> list[float]:
        hist = hist[-self.L:] if hist else []
        if not hist:
            return [0.0] * self.k
        return _forward(self, [self.item(t, E) for t in hist])["F"][-1]

    def attention(self, hist: list[list[str]], E) -> list[float]:
        """The newest step's attention over the history (oldest first): which downloads it leaned on."""
        hist = hist[-self.L:] if hist else []
        if not hist:
            return []
        return _forward(self, [self.item(t, E) for t in hist])["A"][-1]

    @staticmethod
    def score(u: list[float], v: list[float]) -> float:
        return sum(a * b for a, b in zip(u, v))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "SasRec":
        return cls(**json.loads(text))


# ---- small list maths --------------------------------------------------------------------------

def _tower(W, b, x):
    return [math.tanh(bi + sum(w * xj for w, xj in zip(row, x))) for row, bi in zip(W, b)]


def _vm(x, WT):
    """x · W, with W given as its columns (WT = list(zip(*W)))."""
    return [sum(a * c for a, c in zip(x, col)) for col in WT]


def _mv(W, y):
    """W · y (each row of W dotted with y): the backward of x · W."""
    return [sum(a * c for a, c in zip(row, y)) for row in W]


def _outer_add(G, x, y):
    for gi, xi in zip(G, x):
        if xi:
            for j, yj in enumerate(y):
                gi[j] += xi * yj


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _ln(x, g, b):
    n = len(x)
    mu = sum(x) / n
    xc = [v - mu for v in x]
    inv = 1.0 / math.sqrt(sum(v * v for v in xc) / n + EPS)
    xhat = [v * inv for v in xc]
    return [gi * h + bi for gi, h, bi in zip(g, xhat, b)], (xhat, inv)


def _ln_back(dy, g, cache, dg, db):
    xhat, inv = cache
    n = len(dy)
    dxhat = [d * gi for d, gi in zip(dy, g)]
    m1 = sum(dxhat) / n
    m2 = sum(d * h for d, h in zip(dxhat, xhat)) / n
    for j in range(n):
        dg[j] += dy[j] * xhat[j]
        db[j] += dy[j]
    return [inv * (d - m1 - h * m2) for d, h in zip(dxhat, xhat)]


def _softmax(s):
    top = max(s)
    e = [math.exp(v - top) for v in s]
    z = sum(e)
    return [v / z for v in e], top + math.log(z)


# ---- the block -------------------------------------------------------------------------------

def _forward(m: SasRec, X: list[list[float]]) -> dict:
    """The block on one history's item vectors X (oldest first, 1 ≤ n ≤ L)."""
    n, L, sk = len(X), m.L, math.sqrt(m.k)
    WqT, WkT, WvT, W1T, W2T = (list(zip(*W)) for W in (m.wq, m.wk, m.wv, m.w1, m.w2))
    Z = [_add(x, m.p[L - n + t]) for t, x in enumerate(X)]
    Q, K, V = [_vm(z, WqT) for z in Z], [_vm(z, WkT) for z in Z], [_vm(z, WvT) for z in Z]
    A = [_softmax([sum(a * b for a, b in zip(Q[t], K[s])) / sk for s in range(t + 1)])[0] for t in range(n)]
    S, C1, Hp, Hr, F, C2 = [], [], [], [], [], []
    for t in range(n):
        o = [sum(A[t][s] * V[s][j] for s in range(t + 1)) for j in range(m.k)]
        s_, c1 = _ln(_add(Z[t], o), m.g1, m.be1)
        hp = _add(_vm(s_, W1T), m.b1)
        hr = [v if v > 0 else 0.0 for v in hp]
        f, c2 = _ln([a + b + c for a, b, c in zip(s_, _vm(hr, W2T), m.b2)], m.g2, m.be2)
        S.append(s_), C1.append(c1), Hp.append(hp), Hr.append(hr), F.append(f), C2.append(c2)
    return {"Z": Z, "Q": Q, "K": K, "V": V, "A": A, "S": S, "C1": C1, "Hp": Hp, "Hr": Hr, "F": F, "C2": C2}


def _backward(m: SasRec, fw: dict, dF: list[list[float]], g: dict) -> list[list[float]]:
    """Adds the block's gradients into g; returns d(item vector) per history step."""
    Z, Q, K, V, A = fw["Z"], fw["Q"], fw["K"], fw["V"], fw["A"]
    n, L, k, sk = len(Z), m.L, m.k, math.sqrt(m.k)
    dO, dZ = [], []
    for t in range(n):
        dr2 = _ln_back(dF[t], m.g2, fw["C2"][t], g["g2"], g["be2"])
        _outer_add(g["w2"], fw["Hr"][t], dr2)
        for j in range(k):
            g["b2"][j] += dr2[j]
        dhp = [d if h > 0 else 0.0 for d, h in zip(_mv(m.w2, dr2), fw["Hp"][t])]
        _outer_add(g["w1"], fw["S"][t], dhp)
        for j in range(k):
            g["b1"][j] += dhp[j]
        dr1 = _ln_back(_add(dr2, _mv(m.w1, dhp)), m.g1, fw["C1"][t], g["g1"], g["be1"])
        dO.append(dr1)
        dZ.append(list(dr1))
    dQ = [[0.0] * k for _ in range(n)]
    dK = [[0.0] * k for _ in range(n)]
    dV = [[0.0] * k for _ in range(n)]
    for t in range(n):
        a = A[t]
        da = [sum(x * y for x, y in zip(dO[t], V[s])) for s in range(t + 1)]
        mean = sum(ai * di for ai, di in zip(a, da))
        for s in range(t + 1):
            for j in range(k):
                dV[s][j] += a[s] * dO[t][j]
            dl = a[s] * (da[s] - mean) / sk
            if dl:
                for j in range(k):
                    dQ[t][j] += dl * K[s][j]
                    dK[s][j] += dl * Q[t][j]
    for t in range(n):
        _outer_add(g["wq"], Z[t], dQ[t])
        _outer_add(g["wk"], Z[t], dK[t])
        _outer_add(g["wv"], Z[t], dV[t])
        dz = dZ[t]
        for part in (_mv(m.wq, dQ[t]), _mv(m.wk, dK[t]), _mv(m.wv, dV[t])):
            for j in range(k):
                dz[j] += part[j]
        prow = g["p"][L - n + t]
        for j in range(k):
            prow[j] += dz[j]
    return dZ


def _zeros(m: SasRec) -> dict:
    return {f: ([[0.0] * len(r) for r in getattr(m, f)] if f in MATRICES else [0.0] * len(getattr(m, f)))
            for f in FIELDS}


def loss_and_grads(m: SasRec, E, batch: list[Example], lam: float, alpha: float, train_e: bool = False,
                   cache: dict | None = None) -> tuple[float, dict]:
    """Mean loss over the batch + lam·(Σ over MATRICES ‖W‖² [+ Σ over the batch's words ‖E_w‖² when train_e]).

    Each distinct title is pushed through the item tower once per batch, and its gradient is gathered before one
    backward pass. `cache`: title means by token tuple, reused across steps (only valid while E is frozen)."""
    d, k, B = E.dim, m.k, len(batch)
    g = _zeros(m)
    items: dict[tuple, tuple[list[float], list[float]]] = {}  # title -> (h, x)
    dX: dict[tuple, list[float]] = {}

    def x_of(tokens):
        key = tuple(tokens)
        if key not in items:
            if cache is not None and not train_e:
                h = cache.get(key)
                if h is None:
                    h = cache[key] = title_vec(tokens, E)
            else:
                h = title_vec(tokens, E)
            items[key] = (h, _tower(m.wi, m.bi, h))
            dX[key] = [0.0] * k
        return items[key][1]

    def ce(vec, titles, weight):
        """weight · (−log softmax(vec·x)[0]); adds its gradients; returns (loss, d vec)."""
        xs = [x_of(t) for t in titles]
        p, lse = _softmax([sum(a * b for a, b in zip(vec, x)) for x in xs])
        s0 = sum(a * b for a, b in zip(vec, xs[0]))
        dvec = [0.0] * k
        for c, (t, x) in enumerate(zip(titles, xs)):
            ds = weight * (p[c] - (1.0 if c == 0 else 0.0))
            if ds:
                dx = dX[tuple(t)]
                for j in range(k):
                    dvec[j] += ds * x[j]
                    dx[j] += ds * vec[j]
        return weight * (lse - s0), dvec

    total = 0.0
    for ex in batch:
        hist = ex.hist[-m.L:]
        n = len(hist)
        if n:
            fw = _forward(m, [x_of(t) for t in hist])
            u = fw["F"][-1]
        else:
            u = [0.0] * k
        loss, du = ce(u, ex.cands, 1.0 / B)
        total += loss
        if n:
            dF = [[0.0] * k for _ in range(n)]
            dF[-1] = du
            if alpha and n > 1:
                w = alpha / (n - 1) / B
                for t in range(n - 1):
                    negs = ex.seq_negs[t] if t < len(ex.seq_negs) else []
                    loss, df = ce(fw["F"][t], [hist[t + 1]] + list(negs), w)
                    total += loss
                    dF[t] = _add(dF[t], df)
            for t, dz in zip(hist, _backward(m, fw, dF, g)):
                dx = dX[tuple(t)]
                for j in range(k):
                    dx[j] += dz[j]

    # The item tower, once per distinct title.
    gE: dict[str, list[float]] = {}
    for key, (h, x) in items.items():
        dz = [a * (1 - b * b) for a, b in zip(dX[key], x)]
        _outer_add(g["wi"], dz, h)
        for i in range(k):
            g["bi"][i] += dz[i]
        if train_e:
            known = [t for t in key if E.get(t) is not None]
            if known:
                dh = [sum(m.wi[i][j] * dz[i] for i in range(k)) / len(known) for j in range(d)]
                for t in known:
                    row = gE.setdefault(t, [0.0] * d)
                    for j in range(d):
                        row[j] += dh[j]

    reg = 0.0
    for f in MATRICES:
        for grow, wrow in zip(g[f], getattr(m, f)):
            for j, w in enumerate(wrow):
                reg += w * w
                grow[j] += 2 * lam * w
    if train_e:
        used = {t for key in items for t in key if E.get(t) is not None}
        for t in used:
            e = E.get(t)
            reg += sum(x * x for x in e)
            row = gE.setdefault(t, [0.0] * d)
            for j in range(d):
                row[j] += 2 * lam * e[j]
        g["E"] = gE
    return total + lam * reg, g


def step(m: SasRec, vel: dict, grads: dict, lr: float, mu: float, E=None) -> None:
    """SGD with momentum, in place, PyTorch's form (v = μ·v + g; w = w − lr·v). With `E`, word rows move too,
    including rows with no gradient this step: their velocity still carries them, as in PyTorch."""
    for name in FIELDS:
        w, gr = getattr(m, name), grads[name]
        two = name in MATRICES
        v = vel.setdefault(name, [[0.0] * len(r) for r in gr] if two else [0.0] * len(gr))
        for wr, gr_, vr in (zip(w, gr, v) if two else [(w, gr, v)]):
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


def init(d: int, k: int, L: int, rng: random.Random) -> SasRec:
    def mat(r, c):
        return [[rng.gauss(0, 0.1) for _ in range(c)] for _ in range(r)]

    z = [0.0] * k
    return SasRec(wi=mat(k, d), bi=list(z), p=mat(L, k), wq=mat(k, k), wk=mat(k, k), wv=mat(k, k), w1=mat(k, k),
                  b1=list(z), w2=mat(k, k), b2=list(z), g1=[1.0] * k, be1=list(z), g2=[1.0] * k, be2=list(z))


def epochs_for(n_examples: int) -> int:
    """Epochs for a log of this size: EPOCHS while the log is small, then WORK example passes in all (never fewer
    than MIN_EPOCHS), so training after a refresh stays near a minute as the log grows."""
    return max(MIN_EPOCHS, min(EPOCHS, WORK // max(n_examples, 1)))


def fit(examples: list[Example], E, k: int = K, L: int = HISTORY, epochs: int | None = None, seed: int = 0,
        lr: float = LR, mu: float = MU, lam: float = LAMBDA, alpha: float = ALPHA, batch: int = BATCH) -> SasRec:
    """Train everything but E (frozen). Seeded, so the same log gives the same model. `epochs` None: epochs_for."""
    if epochs is None:
        epochs = epochs_for(len(examples))
    rng = random.Random(seed)
    m = init(E.dim, k, L, rng)
    vel: dict = {}
    cache: dict = {}
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            _, g = loss_and_grads(m, E, [examples[i] for i in order[s:s + batch]], lam, alpha, cache=cache)
            step(m, vel, g, lr, mu)
    m.epochs = epochs
    return m


# ---- training examples from the log (design A3) ----------------------------------------------

def examples(rows: list[Row], history: list[tuple[float, list[str]]], rng: random.Random) -> list[Example]:
    """v7's examples (one per download in a titled list, the rest of that list as the "no"s), read in order:
    the history is the newest HISTORY downloads finished before the list, oldest first. For the next-download
    task, step t gets up to NEG_SEQ wrong answers drawn (without replacement) from the videos you were shown and
    didn't download, anywhere in the log, never the download that really came next."""
    hist_sorted = sorted(history, key=lambda h: h[0])
    pool = sorted({tuple(tokenize(r.title)) for r in rows if r.label == 0 and r.title})
    out = []
    for refresh, rs in titled_lists(rows).items():
        hist = [tok for at, tok in hist_sorted if at < refresh][-HISTORY:]
        negs = []
        for t in range(len(hist) - 1):
            options = [list(x) for x in pool if list(x) != hist[t + 1]]
            negs.append(rng.sample(options, min(NEG_SEQ, len(options))))
        negatives = [tokenize(r.title) for r in rs if r.label == 0]
        for r in rs:
            if r.label == 1:
                out.append(Example(hist=hist, cands=[tokenize(r.title)] + negatives, seq_negs=negs))
    return out
