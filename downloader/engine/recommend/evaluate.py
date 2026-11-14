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
from .neural import Net, NetTrainer
from .rank import CHANNEL_BOOST

MIN_ROWS = 200
MIN_DOWNLOADS = 15
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


RANKERS = {"baseline": _baseline, "v1": _v1, "v2": _v2, "v3": _v3}


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
        # Ties go by video id: neutral, and not v1's order, which would favour v1.
        orders = [[r.label for r in sorted(rs, key=lambda r: (-score(r, models), r.video_id))] for rs in judged]
        ndcg[name] = sum(map(ndcg_at, orders)) / len(orders) if orders else 0.0
        hit3[name] = sum(map(hit_at, orders)) / len(orders) if orders else 0.0
    return len(judged), ndcg, hit3


def evaluate(rows: list[Row], trainer, net_trainer: NetTrainer | None = None) -> Report:
    rep = Report(len(rows), sum(r.label for r in rows))
    train, test = split(rows)
    if len({r.label for r in train}) < 2:
        return rep  # nothing to learn from yet: every training row says the same thing
    m = trainer.fit(NAMES, [r.x for r in train], [r.label for r in train], [r.weight for r in train])
    rep.weights = m
    net = (net_trainer or NetTrainer()).fit(NAMES, [r.x for r in train], [r.label for r in train],
                                            [r.weight for r in train])
    rep.net = net
    rep.test_lists, rep.ndcg, rep.hit3 = rank_lists(test, {"v2": m, "v3": net})
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


def choose(rep: Report) -> tuple[str, str]:
    """Which ranker should rank the page: "v1", "v2" or "v3", with the reason in the owner's words (v3 D4).

    v3 must beat v2 outright: on a tie the simpler model wins. Both must at least match v1.
    """
    if rep.rows < MIN_ROWS or rep.downloads < MIN_DOWNLOADS or rep.weights is None or rep.test_lists < MIN_TEST_LISTS:
        return "v1", gate(rep)[1]
    v1, v2, v3 = rep.ndcg["v1"], rep.ndcg["v2"], rep.ndcg.get("v3", 0.0)
    if rep.net is not None and v3 > v2 and v3 >= v1:
        return "v3", f"v3 beats v2 and v1 on your newest lists (NDCG {v3:.2f} vs {v2:.2f} and {v1:.2f})"
    on, why = gate(rep)
    return ("v2", why) if on else ("v1", why)
