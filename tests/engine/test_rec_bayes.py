import math
import random
import statistics

import pytest

from downloader.engine.recommend.bayes import (Posterior, cholesky, fit_posterior, inverse_spd, pick_contextual,
                                               posterior_from_rows)
from downloader.engine.recommend.candidates import Candidate
from downloader.engine.recommend.dataset import Row
from downloader.engine.recommend.features import NAMES, _vector
from downloader.engine.recommend.learn import sigmoid, standardise
from downloader.engine.recommend.rank import Suggestion

NOW = 1_800_000_000.0
NAMES2 = ["signal", "noise"]
_r = random.Random(42)
X = [[_r.uniform(0, 1), _r.uniform(0, 1)] for _ in range(40)]
y = [1 if x[0] + _r.uniform(-0.3, 0.3) > 0.55 else 0 for x in X]
s = [1.0 if i % 3 else 0.3 for i in range(40)]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def grad_at(p, X, y, s):
    """Gradient of Σ sᵢ·lossᵢ + ‖θ‖²/(2σ₀²) at the posterior mean, in standardised units."""
    Z, _, _ = standardise(X)
    g = [th / 1.0 for th in p.mean]
    for z, yi, si in zip(Z, y, s):
        zt = [1.0] + z
        err = si * (sigmoid(sum(a * b for a, b in zip(p.mean, zt))) - yi)
        g = [gj + err * zj for gj, zj in zip(g, zt)]
    return g


def row(label=0, at=NOW, discovery=True, similarity=0.1):
    return Row("v", _vector(similarity, 0.0, 0, 200, "words", 5 if discovery else 1, discovery), label, 1.0, 1.0, at)


def sugg(vid, similarity):
    return Suggestion(Candidate(vid, "u", vid, None, 200, "q", "words"), similarity, similarity, 0.0, None, [])


def test_cholesky_of_a_known_matrix():
    L = cholesky([[4, 2], [2, 3]])
    assert L == [pytest.approx([2, 0]), pytest.approx([1, math.sqrt(2)])]


def test_cholesky_rejects_non_positive_definite():
    with pytest.raises(ValueError):
        cholesky([[1, 2], [2, 1]])


def test_inverse_spd_times_matrix_is_identity():
    A = [[4, 2, 0.5], [2, 3, 0.1], [0.5, 0.1, 2]]
    for got, want in zip(matmul(inverse_spd(A), A), identity(3)):
        assert got == pytest.approx(want, abs=1e-12)


def test_no_rows_is_the_prior():
    p = fit_posterior(["a", "b"], [], [], [])
    assert p.mean == [0, 0, 0] and p.spread() == pytest.approx([1, 1, 1])


def test_gradient_is_zero_at_the_fit():  # the MAP really is the minimum
    p = fit_posterior(NAMES2, X, y, s)
    assert max(abs(g) for g in grad_at(p, X, y, s)) < 1e-6


def test_more_rows_make_it_surer():
    small, big = fit_posterior(NAMES2, X[:10], y[:10], s[:10]), fit_posterior(NAMES2, X, y, s)
    assert all(b < a for a, b in zip(small.spread(), big.spread()))


def test_a_predictive_input_gets_a_positive_weight():  # X[:,0] high ↔ downloaded
    assert fit_posterior(NAMES2, X, y, s).mean[1] > 0.5


def test_all_no_rows_still_fits():
    p = fit_posterior(NAMES2, X, [0] * len(X), s)
    assert p.mean[0] < 0 and all(math.isfinite(v) for v in p.spread())


def test_constant_column_is_safe():
    p = fit_posterior(["c", "x"], [[1.0, v] for v in range(20)], [i % 2 for i in range(20)], [1.0] * 20)
    assert p.mean[1] == pytest.approx(0, abs=1e-9)  # standardised to 0, prior keeps it at 0


def test_samples_spread_like_the_posterior():
    p = fit_posterior(NAMES2, X, y, s)
    draws = [p.sample(random.Random(i))[1] for i in range(4000)]
    assert statistics.stdev(draws) == pytest.approx(p.spread()[1], rel=0.1)


def test_json_round_trip():
    p = fit_posterior(NAMES2, X, y, s)
    assert Posterior.from_json(p.to_json()) == p


def test_posterior_from_rows_uses_discovery_rows_with_fading():
    old_yes, new_no = row(label=1, at=NOW - 300 * 86400), row(label=0)
    non_disc = row(label=1, discovery=False)
    p = posterior_from_rows(NAMES, [old_yes, new_no, non_disc], NOW)
    assert p.rows == pytest.approx(1 + 0.5 ** 10)


def _similarity_posterior():
    rng = random.Random(3)
    rows = []
    for _ in range(80):
        sim = rng.uniform(0, 1)
        rows.append(row(label=1 if sim > 0.5 else 0, similarity=sim))
    return posterior_from_rows(NAMES, rows, NOW)


def test_pick_contextual_takes_the_top_sampled_score():
    p = _similarity_posterior()
    pool = [sugg("low", 0.1), sugg("high", 0.9)]
    wins = sum(pick_contextual(pool, p, random.Random(i)).candidate.video_id == "high" for i in range(100))
    assert wins >= 95


def test_pick_contextual_empty_pool_is_none():
    assert pick_contextual([], _similarity_posterior(), random.Random(0)) is None


def test_map_matches_sklearn():
    lm = pytest.importorskip("sklearn.linear_model")
    p = fit_posterior(NAMES2, X, y, s)
    Z, _, _ = standardise(X)
    # A constant column of 1's stands in for the bias, so it is penalised like the weights.
    model = lm.LogisticRegression(C=1.0, fit_intercept=False, solver="lbfgs", max_iter=10_000, tol=1e-12)
    model.fit([[1.0] + z for z in Z], y, sample_weight=s)
    assert list(model.coef_[0]) == pytest.approx(p.mean, abs=1e-3)
