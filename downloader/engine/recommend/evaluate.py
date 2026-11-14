"""Measure before switching (design v2 C5, v3 D3–D4): do the learned rankers beat v1 on the owner's own lists?

The oldest 80% of lists train the models; on each of the newest 20% that has a download, the videos shown are
put in order four ways (baseline, v1, v2, v3), and each order is scored. Lists are split by time, never
shuffled, so the model can't learn from the future.

- NDCG@10: 1.0 when every download sits at the top of the order; it falls the lower they sit.
- Hit rate@3: how often a download is in the first 3.
- AUC: pick one downloaded and one not-downloaded row; how often does v2 score the downloaded one higher?
- Log-loss: how sure and right v2's probabilities are (lower is better).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .dataset import Row
from .features import NAMES, at_top
from .learn import Weights
from .meaning import v4_score
from .neural import Net, NetTrainer
from .rank import CHANNEL_BOOST

MIN_ROWS = 100  # owner, 2026-11-14: was 200; learning starts sooner, the time-split test still decides
MIN_DOWNLOADS = 10
MIN_TEST_LISTS = 3  # lists with a download in the test part; fewer and the comparison means little
TEST_SHARE = 0.2
K = 10


def ndcg_at(labels: list[int], k: int = K) -> float:
    """`labels` in the order a ranker put them."""
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(labels[:k]))
    ideal = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted(labels, reverse=True)[:k]))
    return dcg / ideal if ideal else 0.0


def hit_at(labels: list[int], k: int = 3) -> int:
    return 1 if any(labels[:k]) else 0


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return 0.5
    right = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return right / (len(pos) * len(neg))


def log_loss(probs: list[float], labels: list[int], weights: list[float]) -> float:
    eps = 1e-12
    total = sum(w * -(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps)))
                for p, y, w in zip(probs, labels, weights))
    return total / sum(weights) if weights else 0.0


def split(rows: list[Row], test_share: float = TEST_SHARE) -> tuple[list[Row], list[Row]]:
    """Whole lists, oldest first: the newest `test_share` of them are the test part."""
    lists = sorted({r.refresh for r in rows})
    n_test = max(1, round(len(lists) * test_share)) if len(lists) > 1 else 0
    cut = set(lists[len(lists) - n_test:])
    return [r for r in rows if r.refresh not in cut], [r for r in rows if r.refresh in cut]


# The three rankers, each a score for a row (higher first).
def _baseline(r: Row, m) -> float:
    return r.x[1]  # favourite channels first


def _v1(r: Row, m) -> float:
    return r.x[0] + CHANNEL_BOOST * r.x[1]  # v1's hand-picked formula


def _v2(r: Row, m) -> float:
    return m["v2"].logit(at_top(r.x))


def _v3(r: Row, m) -> float:
    return m["v3"].logit(at_top(r.x))


def _v4(r: Row, m) -> float:
    return v4_score(r.x[0], r.x[8], r.x[1], m["mix"])  # v1's formula with meaning mixed in


RANKERS = {"baseline": _baseline, "v1": _v1, "v4": _v4, "v2": _v2, "v3": _v3}
SIMPLER_FIRST = ("v1", "v4", "v2", "v3")  # on a tie, the simpler ranker stays


@dataclass
class Report:
    rows: int
    downloads: int
    test_lists: int = 0  # test lists with a download
    ndcg: dict[str, float] = field(default_factory=dict)
    hit3: dict[str, float] = field(default_factory=dict)
    auc: float = 0.5
    log_loss: float = 0.0
    weights: Weights | None = None  # trained on the train part only
    net: Net | None = None  # v3, trained on the train part only
    v3_auc: float = 0.5
    v3_log_loss: float = 0.0


def rank_lists(rows: list[Row], models: dict) -> tuple[int, dict[str, float], dict[str, float]]:
    """Reorder each list in `rows` that has a download with every ranker `models` allows, and score the orders.

    `models` holds the trained "v2" and/or "v3"; rankers without a model are skipped. Returns the number of
    lists judged, and the mean NDCG@10 and hit rate@3 per ranker.
    """
    lists: dict[float, list[Row]] = {}
    for r in rows:
        lists.setdefault(r.refresh, []).append(r)
    judged = [rs for rs in lists.values() if any(r.label for r in rs)]
    ndcg, hit3 = {}, {}
    for name, score in RANKERS.items():
        if name in ("v2", "v3") and name not in models:
            continue
        if name == "v4" and "mix" not in models:
            continue  # no vectors, or meaning off: nothing to measure
        # Ties go by video id: neutral, and not v1's order, which would favour v1.
        orders = [[r.label for r in sorted(rs, key=lambda r: (-score(r, models), r.video_id))] for rs in judged]
        ndcg[name] = sum(map(ndcg_at, orders)) / len(orders) if orders else 0.0
        hit3[name] = sum(map(hit_at, orders)) / len(orders) if orders else 0.0
    return len(judged), ndcg, hit3


def evaluate(rows: list[Row], trainer, net_trainer: NetTrainer | None = None, mix: float | None = None) -> Report:
    """`mix`: v4's MEANING_MIX; None leaves v4 out (no vectors, or "Match by meaning" off)."""
    rep = Report(len(rows), sum(r.label for r in rows))
    train, test = split(rows)
    fixed = {} if mix is None else {"mix": mix}
    if len({r.label for r in train}) < 2:
        # Nothing to learn from yet, but v1 and v4 need no training: they can still be compared.
        rep.test_lists, rep.ndcg, rep.hit3 = rank_lists(test, fixed)
        return rep
    m = trainer.fit(NAMES, [r.x for r in train], [r.label for r in train], [r.weight for r in train])
    rep.weights = m
    net = (net_trainer or NetTrainer()).fit(NAMES, [r.x for r in train], [r.label for r in train],
                                            [r.weight for r in train])
    rep.net = net
    rep.test_lists, rep.ndcg, rep.hit3 = rank_lists(test, {**fixed, "v2": m, "v3": net})
    probs = [m.prob(at_top(r.x)) for r in test]
    rep.auc = auc(probs, [r.label for r in test])
    rep.log_loss = log_loss([m.prob(r.x) for r in test], [r.label for r in test], [r.weight for r in test])
    rep.v3_auc = auc([net.prob(at_top(r.x)) for r in test], [r.label for r in test])
    rep.v3_log_loss = log_loss([net.prob(r.x) for r in test], [r.label for r in test], [r.weight for r in test])
    return rep


def gate(rep: Report) -> tuple[bool, str]:
    """Should v2 rank the page? With the reason, in words the owner sees."""
    if rep.rows < MIN_ROWS or rep.downloads < MIN_DOWNLOADS:
        return False, f"Learning: {rep.rows} of {MIN_ROWS} rows, {rep.downloads} of {MIN_DOWNLOADS} downloads"
    if rep.weights is None or rep.test_lists < MIN_TEST_LISTS:
        return False, (f"Waiting for lists to test on: {rep.test_lists} of {MIN_TEST_LISTS} recent lists "
                       "have a download")
    v1, v2 = rep.ndcg["v1"], rep.ndcg["v2"]
    if v2 < v1:
        return False, f"v2 didn't beat v1 yet (NDCG {v2:.2f} vs {v1:.2f})"
    verb = "beats" if v2 > v1 else "matches"
    return True, f"v2 {verb} v1 on your newest lists (NDCG {v2:.2f} vs {v1:.2f})"


def base(rep: Report, meaning_on: bool, default_on: bool) -> tuple[str, str]:
    """The ranker that needs no training: v4 or v1 (design v4, E7), with the reason in the owner's words.

    v4 when "Match by meaning" is on and either it beat v1 on the simulated months (`default_on`) or it beats
    v1 on the owner's own newest lists. Once those lists exist they decide, also against the starter's check.
    """
    if not meaning_on:
        return "v1", "Match by meaning is off"
    own = rep.test_lists >= MIN_TEST_LISTS and "v4" in rep.ndcg
    v1, v4 = rep.ndcg.get("v1", 0.0), rep.ndcg.get("v4", 0.0)
    if own and v4 > v1:
        return "v4", f"v4 beats v1 on your newest lists (NDCG {v4:.2f} vs {v1:.2f})"
    if own and v4 < v1:
        return "v1", f"v4 didn't beat v1 on your newest lists (NDCG {v4:.2f} vs {v1:.2f})"
    if default_on:
        return "v4", "Meaning (v4) is on: it beat v1 on the simulated months"
    return "v1", "v4 hasn't beaten v1 on your lists yet"


def choose(rep: Report, meaning_on: bool = False, default_on: bool = False) -> tuple[str, str]:
    """Which ranker should rank the page: "v1", "v4", "v2" or "v3", with the reason in the owner's words.

    The base (v1 or v4, see `base`) ranks until the learned ones pass the size gate. Then v2 must at least
    match the base, and v3 must beat v2 outright and match the base: on a tie the simpler model wins.
    """
    first, first_why = base(rep, meaning_on, default_on)
    if rep.rows < MIN_ROWS or rep.downloads < MIN_DOWNLOADS or rep.weights is None or rep.test_lists < MIN_TEST_LISTS:
        return first, gate(rep)[1]
    b, v2, v3 = rep.ndcg.get(first, 0.0), rep.ndcg["v2"], rep.ndcg.get("v3", 0.0)
    if rep.net is not None and v3 > v2 and v3 >= b:
        return "v3", f"v3 beats v2 and {first} on your newest lists (NDCG {v3:.2f} vs {v2:.2f} and {b:.2f})"
    if first == "v1":
        on, why = gate(rep)
        return ("v2", why) if on else ("v1", why)
    if v2 >= b:
        verb = "beats" if v2 > b else "matches"
        return "v2", f"v2 {verb} v4 on your newest lists (NDCG {v2:.2f} vs {b:.2f})"
    return "v4", f"{first_why}; v2 didn't beat it yet (NDCG {v2:.2f} vs {b:.2f})"
