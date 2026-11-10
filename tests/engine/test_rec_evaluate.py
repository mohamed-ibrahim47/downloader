import math

import pytest

from downloader.engine.recommend.dataset import Row
from downloader.engine.recommend.evaluate import (MIN_DOWNLOADS, MIN_ROWS, auc, evaluate, gate, hit_at, log_loss,
                                                  ndcg_at, split)
from downloader.engine.recommend.learn import HandTrainer


def _row(refresh, sim, aff, label, vid=None, rank=1):
    x = [sim, aff, 0, 1.0, 0, 0, math.log(rank), 0]
    return Row(vid or f"{refresh}-{sim}-{aff}", x, label, 1.0 if label else 0.3, float(refresh), float(refresh))


def test_ndcg_by_hand():
    assert ndcg_at([1, 0, 0], 10) == 1.0
    # The only download at place 2: DCG = 1/log2(3); the best order has it at place 1: IDCG = 1.
    assert ndcg_at([0, 1, 0], 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at([0, 0, 0], 10) == 0.0
    assert ndcg_at([0, 0, 1], 2) == 0.0  # past the cut-off


def test_hit_rate():
    assert hit_at([0, 0, 1], 3) == 1 and hit_at([0, 0, 0, 1], 3) == 0


def test_auc_counts_correctly_ordered_pairs():
    assert auc([0.9, 0.1], [1, 0]) == 1.0 and auc([0.1, 0.9], [1, 0]) == 0.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5  # a tie is half right


def test_log_loss_is_weighted():
    assert log_loss([0.5], [1], [1.0]) == pytest.approx(math.log(2))


def test_split_keeps_time_order():
    rows = [_row(r, 0.1, 0, 0) for r in range(10)]
    train, test = split(rows)
    assert max(r.refresh for r in train) < min(r.refresh for r in test)
    assert len({r.refresh for r in test}) == 2  # the newest 20% of lists


def _world(n_lists=40, per_list=10):
    """The owner downloads by channel, not by words: a world where v1's 0.15 is far too small."""
    rows = []
    for r in range(n_lists):
        for i in range(per_list):
            aff = 1.0 if i == (r % per_list) else 0.0
            sim = 0.9 if i == ((r + 5) % per_list) else 0.1  # a strong word match the owner ignores
            rows.append(_row(r, sim, aff, 1 if aff == 1.0 else 0, rank=i + 1))
    return rows


def test_v2_learns_what_v1_got_wrong():
    rep = evaluate(_world(), HandTrainer())
    assert rep.test_lists >= 3
    assert rep.ndcg["v2"] > rep.ndcg["v1"] and rep.ndcg["v2"] == pytest.approx(1.0)
    assert rep.ndcg["baseline"] == pytest.approx(1.0)  # here, favourite channels first is the right answer
    assert rep.auc > 0.9 and rep.weights is not None


def test_the_gate():
    world = _world()
    on, why = gate(evaluate(world, HandTrainer()))
    assert on and "beats" in why

    few = world[:MIN_ROWS - 1]
    on, why = gate(evaluate(few, HandTrainer()))
    assert not on and f"of {MIN_ROWS} rows" in why

    no_yes = [_row(r.refresh, r.x[0], r.x[1], 0) for r in world]
    on, why = gate(evaluate(no_yes, HandTrainer()))
    assert not on and f"0 of {MIN_DOWNLOADS} downloads" in why


def test_the_gate_keeps_v1_when_v2_loses():
    # The owner follows words, which v1 already ranks by; v2 has nothing to learn but noise from rank.
    rows = []
    for r in range(40):
        for i in range(10):
            sim = 0.9 if i == r % 10 else 0.1
            rows.append(_row(r, sim, 0.0, 1 if sim == 0.9 else 0, rank=i + 1))
    rep = evaluate(rows, HandTrainer())
    rep.ndcg["v2"] = rep.ndcg["v1"] - 0.01  # force a loss, whatever the data
    on, why = gate(rep)
    assert not on and "didn't beat" in why


def test_one_label_only_in_training_does_not_crash():
    rows = [_row(r, 0.1, 0, 0) for r in range(5)] + [_row(9, 0.9, 0, 1)]
    rep = evaluate(rows, HandTrainer())
    assert rep.weights is None and rep.test_lists <= 1
