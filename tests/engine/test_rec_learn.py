import math
import random

import pytest

from downloader.engine.recommend.learn import HandTrainer, Weights, loss_and_grad, sigmoid, standardise


def _synthetic(n=2000, w=(2.0, -1.0), b=-0.5, seed=1):
    """Rows drawn from a known logistic model, so we know the answer the trainer should find."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x = [rng.gauss(0, 1), rng.gauss(0, 1)]
        p = sigmoid(b + w[0] * x[0] + w[1] * x[1])
        X.append(x)
        y.append(1 if rng.random() < p else 0)
    return X, y


def test_sigmoid_is_safe_for_big_numbers():
    assert sigmoid(0) == 0.5 and sigmoid(1000) == 1.0 and sigmoid(-1000) == pytest.approx(0.0)


def test_standardise_gives_mean_zero_spread_one_and_survives_a_constant_column():
    Z, means, stds = standardise([[1.0, 5.0], [3.0, 5.0]])
    assert means == [2.0, 5.0] and stds == [1.0, 1.0]  # a constant column gets spread 1, not 0
    assert Z == [[-1.0, 0.0], [1.0, 0.0]]


def test_the_gradient_matches_a_numeric_one():
    # The classic check for a hand-written gradient: nudge each weight a tiny bit and watch the loss.
    rng = random.Random(3)
    Z = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(20)]
    y = [rng.randint(0, 1) for _ in range(20)]
    s = [rng.choice([1.0, 0.3]) for _ in range(20)]
    w, b, lam, eps = [0.3, -0.2, 0.1], 0.05, 0.01, 1e-6
    _, gw, gb = loss_and_grad(Z, y, s, w, b, lam)
    for j in range(3):
        up = w[:]; up[j] += eps
        down = w[:]; down[j] -= eps
        numeric = (loss_and_grad(Z, y, s, up, b, lam)[0] - loss_and_grad(Z, y, s, down, b, lam)[0]) / (2 * eps)
        assert gw[j] == pytest.approx(numeric, abs=1e-6)
    numeric_b = (loss_and_grad(Z, y, s, w, b + eps, lam)[0] - loss_and_grad(Z, y, s, w, b - eps, lam)[0]) / (2 * eps)
    assert gb == pytest.approx(numeric_b, abs=1e-6)


def test_the_trainer_finds_the_weights_that_made_the_data():
    X, y = _synthetic()
    m = HandTrainer(lam=0.0, epochs=2000).fit(["a", "b"], X, y, [1.0] * len(y))
    # Back from standardised units to the data's own units, to compare with the truth.
    raw = [wj / sj for wj, sj in zip(m.w, m.stds)]
    assert raw == pytest.approx([2.0, -1.0], abs=0.25)
    assert m.prob([0.0, 0.0]) == pytest.approx(sigmoid(-0.5), abs=0.05)


def test_heavier_rows_count_more():
    X = [[1.0], [1.0]]
    light = HandTrainer(lam=0.0).fit(["a"], X * 50, [1, 0] * 50, [1.0, 0.3] * 50)
    assert light.prob([1.0]) > 0.7  # the "no" rows weigh 0.3, so the "yes" wins: 1 / 1.3 ≈ 0.77


def test_l2_pulls_weights_towards_zero():
    X, y = _synthetic(n=300)
    free = HandTrainer(lam=0.0).fit(["a", "b"], X, y, [1.0] * len(y))
    tied = HandTrainer(lam=1.0).fit(["a", "b"], X, y, [1.0] * len(y))
    assert sum(map(abs, tied.w)) < sum(map(abs, free.w))


def test_weights_survive_a_round_trip_through_json():
    m = Weights(["a"], [0.5], -0.1, [1.0], [2.0], "hand")
    assert Weights.from_json(m.to_json()) == m
    assert m.logit([3.0]) == pytest.approx(-0.1 + 0.5 * (3.0 - 1.0) / 2.0)


def test_training_is_deterministic():
    X, y = _synthetic(n=200)
    a = HandTrainer().fit(["a", "b"], X, y, [1.0] * 200)
    b = HandTrainer().fit(["a", "b"], X, y, [1.0] * 200)
    assert a == b and not math.isnan(a.b)


def test_hand_is_always_offered_and_sklearn_only_when_it_imports(monkeypatch):
    from downloader.engine.recommend import learn
    assert "hand" in learn.available_trainers()
    monkeypatch.setattr(learn, "_sklearn", lambda: None)
    assert learn.available_trainers() == ["hand"]
    assert learn.trainer("sklearn").name == "hand"  # asked for but missing: fall back, don't crash


def test_both_trainers_find_the_same_weights():
    # Same objective, and it has one minimum (it's convex), so both must land on the same answer.
    pytest.importorskip("sklearn")
    from downloader.engine.recommend.learn import SklearnTrainer
    X, y = _synthetic(n=600)
    s = [1.0 if yi else 0.3 for yi in y]
    hand = HandTrainer(epochs=5000, tolerance=1e-12).fit(["a", "b"], X, y, s)
    sk = SklearnTrainer().fit(["a", "b"], X, y, s)
    assert sk.trainer == "sklearn"
    assert sk.w == pytest.approx(hand.w, abs=0.01) and sk.b == pytest.approx(hand.b, abs=0.01)
