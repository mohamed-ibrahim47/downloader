import json
import math
from pathlib import Path

import pytest

from downloader.engine.recommend import twotower
from downloader.engine.recommend.twotower import Example, TwoTower

CASE = Path(__file__).parent / "fixtures" / "two_tower_case.json"


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
    m = TwoTower(wi=case["Wi"], bi=case["bi"], wu=case["Wu"], bu=case["bu"])
    batch = [Example(hist=ex["hist"], cands=ex["cands"]) for ex in case["examples"]]
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
    loss, g = twotower.loss_and_grads(m, E, batch, case["lam"], train_e=True)
    assert abs(loss - case["loss"]) <= 1e-12
    want = case["grads"]
    for mine, theirs in (("wi", "Wi"), ("bi", "bi"), ("wu", "Wu"), ("bu", "bu")):
        _close(g[mine], want[theirs])
    assert set(g["E"]) == set(want["E"])
    for w, row in want["E"].items():
        _close(g["E"][w], row)


def test_fixture_two_steps():
    case, E, m, batch = _case()
    vel: dict = {}
    for ex in batch:  # one example per step, as the lab did
        _, g = twotower.loss_and_grads(m, E, [ex], case["lam"], train_e=True)
        twotower.step(m, vel, g, case["lr"], case["mu"], E=E)
    after = case["after_2_steps"]
    for mine, theirs in (("wi", "Wi"), ("bi", "bi"), ("wu", "Wu"), ("bu", "bu")):
        _close(getattr(m, mine), after[theirs])
    for w, row in after["E"].items():
        _close(E.get(w), row)


def test_unknown_title_scores_finite():
    _, E, m, _ = _case()
    u = m.user([], E)
    s = m.score(u, m.item(["nothing", "known"], E))
    assert math.isfinite(s)
    assert m.title_vec([], E) == [0.0] * E.dim


def _toy():
    E = Words({"cat": [1.0, 0.0], "kitten": [0.9, 0.1], "car": [0.0, 1.0], "engine": [0.1, 0.9]})
    cat, car = ["cat", "kitten"], ["car", "engine"]
    ex = [Example(hist=[cat, ["cat"]], cands=[["kitten"], ["car"]]),
          Example(hist=[car, ["car"]], cands=[["engine"], ["cat"]])] * 10
    return E, ex, cat, car


def test_fit_frozen_e_learns():
    E, ex, cat, car = _toy()
    before = json.dumps(E.table)
    m = twotower.fit(ex, E, k=4, epochs=60, seed=1)
    u_cat = m.user([cat], E)
    assert m.score(u_cat, m.item(["cat"], E)) > m.score(u_cat, m.item(["car"], E))
    u_car = m.user([car], E)
    assert m.score(u_car, m.item(["engine"], E)) > m.score(u_car, m.item(["kitten"], E))
    assert json.dumps(E.table) == before  # the word vectors are frozen


def test_json_roundtrip():
    E, ex, cat, _ = _toy()
    m = twotower.fit(ex, E, k=4, epochs=2, seed=1)
    m.vectors_at = 123.0
    back = TwoTower.from_json(m.to_json())
    assert back.vectors_at == 123.0
    u = m.user([cat], E)
    assert back.score(back.user([cat], E), back.item(["car"], E)) == pytest.approx(m.score(u, m.item(["car"], E)))


def _row(vid, refresh, label, title):
    from downloader.engine.recommend.dataset import Row
    return Row(vid, [], label, 1.0, refresh, refresh, title)


def test_examples_two_downloads_one_list():
    rs = [_row("a", 10.0, 1, "cat one"), _row("b", 10.0, 1, "cat two"), _row("c", 10.0, 0, "car three")]
    ex = twotower.examples(rs, [])
    assert [e.cands[0] for e in ex] == [["cat", "one"], ["cat", "two"]]
    assert ["cat", "two"] in ex[0].cands[1:] and ["cat", "one"] in ex[1].cands[1:]
    assert all(len(e.cands) == 3 for e in ex)


def test_examples_skip_untitled_list():
    rs = [_row("a", 10.0, 1, "cat one"), _row("b", 10.0, 0, None), _row("c", 20.0, 1, "car")]
    assert list(twotower.titled_lists(rs)) == [20.0]
    assert [e.cands[0] for e in twotower.examples(rs, [])] == [["car"]]


def test_examples_history_before_refresh_only():
    history = [(float(i), [f"w{i}"]) for i in range(60)] + [(99.0, ["later"])]
    ex = twotower.examples([_row("a", 70.0, 1, "cat"), _row("b", 70.0, 0, "car")], history)[0]
    assert ["later"] not in ex.hist
    assert len(ex.hist) == twotower.HISTORY and ex.hist[-1] == ["w59"] and ex.hist[0] == ["w10"]


def test_examples_empty_history():
    ex = twotower.examples([_row("a", 5.0, 1, "cat"), _row("b", 5.0, 0, "car")], [(9.0, ["later"])])
    assert len(ex) == 1 and ex[0].hist == []


def test_history_of_downloads_only():
    from downloader.engine.recommend.profile import Profile, ProfileDoc
    p = Profile([ProfileDoc("v1", "Cat", ["cat"], 1.0, added_at=5.0), ProfileDoc("v2", "Car", ["car"], -0.5)], {})
    assert twotower.history_of(p) == [(5.0, ["cat"])]
