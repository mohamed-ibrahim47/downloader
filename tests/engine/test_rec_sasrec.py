import json
import math
import random
from pathlib import Path

from downloader.engine.recommend import sasrec
from downloader.engine.recommend.sasrec import Example, SasRec

CASE = Path(__file__).parent / "fixtures" / "sasrec_case.json"
NAMES = {"wi": "Wi", "bi": "bi", "p": "P", "wq": "Wq", "wk": "Wk", "wv": "Wv", "w1": "W1", "b1": "b1", "w2": "W2",
         "b2": "b2", "g1": "g1", "be1": "be1", "g2": "g2", "be2": "be2"}  # app field -> lab field


class Words:
    """A word table with the same face as embed.Vectors: .get(word) and .dim."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = {w: list(v) for w, v in table.items()}
        self.dim = len(next(iter(self.table.values())))

    def get(self, word):
        return self.table.get(word)


def _case():
    case = json.loads(CASE.read_text(encoding="utf-8"))
    E = Words(dict(zip(case["words"], case["E"])))
    m = SasRec(**{mine: case[lab] for mine, lab in NAMES.items()})
    batch = [Example(hist=ex["hist"], cands=ex["cands"], seq_negs=ex["seq_negs"]) for ex in case["examples"]]
    return case, E, m, batch


def _close(a, b, tol=1e-12):
    if isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y, tol)
    else:
        assert abs(a - b) <= tol, (a, b)


def test_fixture_loss_and_grads():
    case, E, m, batch = _case()
    assert m.L == case["L"] == 4
    loss, g = sasrec.loss_and_grads(m, E, batch, case["lam"], case["alpha"], train_e=True)
    assert abs(loss - case["loss"]) <= 1e-12
    for mine, lab in NAMES.items():
        _close(g[mine], case["grads"][lab])
    assert set(g["E"]) == set(case["grads"]["E"])
    for w, row in case["grads"]["E"].items():
        _close(g["E"][w], row)


def test_fixture_two_steps():
    case, E, m, batch = _case()
    vel: dict = {}
    for ex in batch:  # one example per step, as the lab did
        _, g = sasrec.loss_and_grads(m, E, [ex], case["lam"], case["alpha"], train_e=True)
        sasrec.step(m, vel, g, case["lr"], case["mu"], E=E)
    after = case["after_2_steps"]
    for mine, lab in NAMES.items():
        _close(getattr(m, mine), after[lab])
    for w, row in after["E"].items():
        _close(E.get(w), row)


def test_short_history_alone_equals_in_batch():
    case, E, m, batch = _case()
    _, both = sasrec.loss_and_grads(m, E, batch, 0.0, case["alpha"], train_e=True)
    _, long = sasrec.loss_and_grads(m, E, batch[:1], 0.0, case["alpha"], train_e=True)
    _, short = sasrec.loss_and_grads(m, E, batch[1:], 0.0, case["alpha"], train_e=True)
    assert len(batch[1].hist) == 2 < len(batch[0].hist)

    def flat(x):
        return [v for r in x for v in (r if isinstance(r, list) else [r])]

    for name in NAMES:
        for b, lg, sg in zip(flat(both[name]), flat(long[name]), flat(short[name])):
            assert abs(2 * b - lg - sg) <= 1e-12


def test_edge_histories_score_finite():
    case, E, m, _ = _case()
    assert m.user([], E) == [0.0] * m.k
    assert m.score(m.user([], E), m.item(["zorp"], E)) == 0.0
    u = m.user([["nothing", "known"]], E)
    v = m.item(["blim", "mup"], E)
    assert all(math.isfinite(x) for x in u) and math.isfinite(m.score(u, v))


def test_fit_frozen_e_learns_order():
    E = Words({"cat": [1.0, 0.0, 0.3, 0.0], "car": [0.0, 1.0, 0.0, 0.3], "the": [0.2, 0.2, 0.2, 0.2]})
    cat, car = ["the", "cat"], ["the", "car"]
    ends_cat, ends_car = [car, cat, car, cat], [cat, car, cat, car]  # the same titles, another order
    ex = [Example(hist=ends_cat, cands=[cat, car]), Example(hist=ends_car, cands=[car, cat])] * 20
    before = json.dumps(E.table)
    m = sasrec.fit(ex, E, k=4, L=4, epochs=60, seed=1, alpha=0.0)
    assert json.dumps(E.table) == before
    u_cat, u_car = m.user(ends_cat, E), m.user(ends_car, E)
    assert m.score(u_cat, m.item(cat, E)) > m.score(u_cat, m.item(car, E))
    assert m.score(u_car, m.item(car, E)) > m.score(u_car, m.item(cat, E))


def test_attention_row_sums_to_one():
    case, E, m, batch = _case()
    a = m.attention(batch[0].hist, E)
    assert len(a) == 4 and abs(sum(a) - 1) < 1e-12


def test_json_roundtrip():
    case, E, m, batch = _case()
    m.vectors_at = 12.5
    back = SasRec.from_json(m.to_json())
    u, u2 = m.user(batch[0].hist, E), back.user(batch[0].hist, E)
    assert back.vectors_at == 12.5
    assert m.score(u, m.item(["quax"], E)) == back.score(u2, back.item(["quax"], E))


def test_init_shapes():
    m = sasrec.init(d=32, k=16, L=50, rng=random.Random(0))
    assert (len(m.wi), len(m.wi[0]), len(m.p), len(m.wq), m.k, m.L) == (16, 32, 50, 16, 16, 50)
    assert m.g1 == [1.0] * 16 and m.be2 == [0.0] * 16


def test_epochs_follow_a_work_budget():
    assert sasrec.HISTORY == 20
    assert sasrec.epochs_for(10) == sasrec.EPOCHS == 40  # a small log trains fully
    assert sasrec.epochs_for(300) == 5  # 1,500 example passes: about a minute at history 20
    assert sasrec.epochs_for(5000) == sasrec.MIN_EPOCHS == 5


# ---- training examples from the log (Task 5) ----------------------------------------------

from downloader.engine.recommend import twotower  # noqa: E402
from downloader.engine.recommend.dataset import Row  # noqa: E402


def _row(vid, refresh, label, title):
    return Row(vid, [], label, 1.0, refresh, refresh, title)


def _log():
    """Two lists. List 10: a downloaded, b and c skipped. List 20: d downloaded, e skipped, f untitled-free."""
    return [_row("a", 10.0, 1, "alpha song"), _row("b", 10.0, 0, "beta clip"), _row("c", 10.0, 0, "gamma talk"),
            _row("d", 20.0, 1, "delta tune"), _row("e", 20.0, 0, "epsilon show"), _row("f", 20.0, 0, "zeta film")]


HIST = [(3.0, ["three"]), (1.0, ["one"]), (2.0, ["two"]), (15.0, ["fifteen"]), (25.0, ["later"])]


def test_examples_match_twotower_lists():
    rows = _log()
    mine = sasrec.examples(rows, HIST, random.Random(0))
    theirs = twotower.examples(rows, HIST)
    assert [e.cands for e in mine] == [e.cands for e in theirs]


def test_examples_history_ordered_oldest_first():
    ex = sasrec.examples(_log(), HIST, random.Random(0))
    assert ex[0].hist == [["one"], ["two"], ["three"]]


def test_examples_history_before_refresh_only():
    ex = sasrec.examples(_log(), HIST, random.Random(0))
    assert ex[1].hist == [["one"], ["two"], ["three"], ["fifteen"]]  # "later" (25.0) is after list 20
    many = [(i * 0.1, [f"w{i}"]) for i in range(30)]  # all 30 before list 10
    long = sasrec.examples(_log(), many, random.Random(0))[0]
    assert len(long.hist) == sasrec.HISTORY and long.hist[-1] == ["w29"]


def test_seq_negs_from_not_downloaded_titles():
    rows = _log()
    shown_no = {tuple(twotower.tokenize(r.title)) for r in rows if r.label == 0}
    for e in sasrec.examples(rows, HIST, random.Random(0)):
        assert len(e.seq_negs) == max(len(e.hist) - 1, 0)
        for t, negs in enumerate(e.seq_negs):
            assert 0 < len(negs) <= sasrec.NEG_SEQ
            assert len({tuple(n) for n in negs}) == len(negs)  # without replacement
            for n in negs:
                assert tuple(n) in shown_no and n != e.hist[t + 1]


def test_seq_negs_few_titles():
    rows = [_row("a", 10.0, 1, "alpha song"), _row("b", 10.0, 0, "beta clip"), _row("c", 10.0, 0, "gamma talk")]
    ex = sasrec.examples(rows, HIST, random.Random(0))
    assert all(len(n) <= 2 for n in ex[0].seq_negs)
    lone = [_row("a", 10.0, 1, "alpha song"), _row("a2", 10.0, 1, "second song")]  # nothing was skipped
    ex2 = sasrec.examples(lone, HIST, random.Random(0))
    assert ex2 and all(n == [] for e in ex2 for n in e.seq_negs)
    E = Words({"one": [0.1, 0.2], "two": [0.3, -0.1], "three": [0.0, 0.5], "song": [0.2, 0.2]})
    sasrec.fit(ex2, E, k=2, epochs=2)  # no next-download negatives: still trains


def test_examples_empty_history():
    ex = sasrec.examples(_log(), [], random.Random(0))
    assert len(ex) == 2 and ex[0].hist == [] and ex[0].seq_negs == []
