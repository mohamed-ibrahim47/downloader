import random

import pytest

from downloader.engine.recommend.evaluate import auc
from downloader.engine.recommend.learn import HandTrainer
from downloader.engine.recommend.neural import ACTIVATIONS, Net, NetTrainer, init_net, loss_and_grads


def _xor(n=400, seed=2):
    """Yes when exactly one of two inputs is high: no single weight per input can express that."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        a, b = rng.randint(0, 1), rng.randint(0, 1)
        X.append([a + rng.gauss(0, 0.1), b + rng.gauss(0, 0.1)])
        y.append(a ^ b)
    return X, y


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_backprop_matches_a_numeric_gradient(activation):
    # Nudge every weight a tiny bit each way and watch the loss: backprop must agree with that slope.
    rng = random.Random(5)
    Z = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(15)]
    y = [rng.randint(0, 1) for _ in range(15)]
    s = [rng.choice([1.0, 0.3]) for _ in range(15)]
    net = init_net(["a", "b", "c"], 4, activation, random.Random(1))
    lam, eps = 0.01, 1e-6
    _, g = loss_and_grads(net, Z, y, s, lam)

    def loss_with(path, value):
        n = Net.from_json(net.to_json())
        target = n
        for key in path[:-1]:
            target = getattr(target, key) if isinstance(key, str) else target[key]
        if isinstance(path[-1], str):
            setattr(target, path[-1], value)
        else:
            target[path[-1]] = value
        return loss_and_grads(n, Z, y, s, lam)[0]

    checks = [(("W1", i, j), net.W1[i][j], g["W1"][i][j]) for i in range(4) for j in range(3)]
    checks += [(("b1", i), net.b1[i], g["b1"][i]) for i in range(4)]
    checks += [(("w2", i), net.w2[i], g["w2"][i]) for i in range(4)]
    checks += [(("b2",), net.b2, g["b2"])]
    for path, value, grad in checks:
        numeric = (loss_with(path, value + eps) - loss_with(path, value - eps)) / (2 * eps)
        assert grad == pytest.approx(numeric, abs=1e-6), path


def test_a_hidden_layer_learns_xor_and_logistic_regression_cannot():
    X, y = _xor()
    flat = HandTrainer(lam=0.0).fit(["a", "b"], X, y, [1.0] * len(y))
    net = NetTrainer(max_epochs=300, patience=300).fit(["a", "b"], X, y, [1.0] * len(y))
    assert auc([flat.prob(x) for x in X], y) < 0.65  # a straight line can't split XOR
    assert auc([net.prob(x) for x in X], y) > 0.95


def test_same_rows_same_net():
    X, y = _xor(100)
    a = NetTrainer(max_epochs=20).fit(["a", "b"], X, y, [1.0] * len(y))
    b = NetTrainer(max_epochs=20).fit(["a", "b"], X, y, [1.0] * len(y))
    assert a.to_json() == b.to_json()


def test_json_round_trip_scores_the_same():
    X, y = _xor(100)
    net = NetTrainer(max_epochs=20).fit(["a", "b"], X, y, [1.0] * len(y))
    back = Net.from_json(net.to_json())
    assert back.prob([1.0, 0.0]) == net.prob([1.0, 0.0])
    assert back.size() == 2 * 8 + 8 + 8 + 1


def test_early_stopping_stops_on_noise_and_keeps_the_best_pass():
    rng = random.Random(9)
    X = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(200)]
    y = [rng.randint(0, 1) for _ in range(200)]  # pure noise: the held-back loss can only get worse
    net = NetTrainer(max_epochs=200, patience=10).fit(list("abcd"), X, y, [1.0] * 200)
    assert 1 <= net.epochs < 60


def test_a_fixed_number_of_passes_uses_all_rows():
    X, y = _xor(100)
    net = NetTrainer(epochs=7).fit(["a", "b"], X, y, [1.0] * len(y))
    assert net.epochs == 7
