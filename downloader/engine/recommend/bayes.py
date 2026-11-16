"""The contextual bandit: Bayesian logistic regression, written by hand (design v5, V3).

The model is v2's,  P(download) = sigmoid(b + w · z)  over the standardised inputs z, but it keeps a
**belief over the weights** instead of one value for each. Thompson sampling then draws one set of weights
from that belief per pick and lets them choose the card: weights it is unsure about vary a lot between
draws, so cards they favour get tried; weights it is sure about barely move.

**Training (the Laplace approximation).** The exact belief has no closed form, so it is approximated by a
normal distribution centred on the most likely weights:

1. Prior: each weight and the bias ~ N(0, σ₀²), σ₀ = 1 (inputs are standardised, so 1 is "a big effect").
2. The most likely weights θ = [b, w] (the MAP) minimise

       E(θ) = Σ sᵢ · lossᵢ  +  ‖θ‖² / (2σ₀²)

   which is v2's penalised log-loss (with the bias penalised too, because it has a prior).
3. The curvature there, the Hessian  H = Σ sᵢ pᵢ(1 − pᵢ) z̃ᵢ z̃ᵢᵀ + I/σ₀²  (z̃ = [1, z]), says how sure:
   a sharp minimum is a sure belief. The belief is ≈ N(θ, H⁻¹).
4. To draw from N(θ, H⁻¹), keep the Cholesky factor L of H⁻¹ (H⁻¹ = L·Lᵀ): θ + L·ε with ε ~ N(0, I) has
   exactly that spread.

**Why Newton's method** finds the MAP here, not v2's gradient descent: the Hessian is needed anyway for the
uncertainty, and with it each step  θ ← θ − H⁻¹·g  jumps almost straight to the minimum (a handful of steps
instead of hundreds). It works because E is convex: one minimum, and H is always positive definite (the
I/σ₀² term makes sure of that, even with a constant input or rows that are all "no").

Matrices are 10×10 at most, so lists of lists and plain loops are plenty.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass

from .bandit import discovery_rows, fade
from .dataset import Row
from .features import from_suggestion
from .learn import sigmoid, standardise
from .rank import CHANNEL_BOOST, Suggestion

SIGMA0 = 1.0
NEWTON_STEPS = 50
NEWTON_TOL = 1e-8


# ---- small linear algebra -------------------------------------------------------------------------------

def cholesky(A: list[list[float]]) -> list[list[float]]:
    """The lower-triangular L with L·Lᵀ = A. Raises ValueError if A isn't symmetric positive definite.

    Row by row:  L[i][i] = √(A[i][i] − Σₖ L[i][k]²),   L[i][j] = (A[i][j] − Σₖ L[i][k]·L[j][k]) / L[j][j].
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if acc <= 0:
                    raise ValueError("matrix is not positive definite")
                L[i][i] = math.sqrt(acc)
            else:
                L[i][j] = acc / L[j][j]
    return L


def _solve_lower(L, b):
    """x with L·x = b (forward substitution)."""
    x = []
    for i in range(len(L)):
        x.append((b[i] - sum(L[i][k] * x[k] for k in range(i))) / L[i][i])
    return x


def _solve_upper_t(L, b):
    """x with Lᵀ·x = b (back substitution, reading Lᵀ from L)."""
    n = len(L)
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def _solve_spd(L, b):
    """x with A·x = b, given A's Cholesky factor L: solve L·y = b, then Lᵀ·x = y."""
    return _solve_upper_t(L, _solve_lower(L, b))


def inverse_spd(A: list[list[float]]) -> list[list[float]]:
    """A⁻¹ for a symmetric positive definite A, one column of the identity at a time."""
    L = cholesky(A)
    n = len(A)
    cols = [_solve_spd(L, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ---- the belief ----------------------------------------------------------------------------------------

@dataclass
class Posterior:
    """The belief over [bias, weights]: N(mean, chol·cholᵀ). Weights are in standardised units."""
    names: list[str]
    mean: list[float]  # mean[0] is the bias
    chol: list[list[float]]  # Cholesky factor of the covariance H⁻¹, size len(names) + 1
    means: list[float]  # how to standardise raw inputs, as in learn.Weights
    stds: list[float]
    rows: float  # how many rows it learned from (faded weights summed)

    def sample(self, rng: random.Random) -> list[float]:
        """One draw θ̃ = mean + L·ε, ε ~ N(0, I)."""
        eps = [rng.gauss(0.0, 1.0) for _ in self.mean]
        return [m + sum(self.chol[i][k] * eps[k] for k in range(i + 1)) for i, m in enumerate(self.mean)]

    def spread(self) -> list[float]:
        """Each entry's standard deviation: √ of the covariance diagonal (row i of L, squared and summed)."""
        return [math.sqrt(sum(v * v for v in row)) for row in self.chol]

    def score(self, theta: list[float], x: list[float]) -> float:
        """The logit b + w·z of raw inputs x under weights theta."""
        return theta[0] + sum(t * (xj - m) / sd for t, xj, m, sd in zip(theta[1:], x, self.means, self.stds))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Posterior":
        return cls(**json.loads(text))


def _objective(theta, Zt, y, s, prec):
    total = prec / 2 * sum(t * t for t in theta)
    for z, yi, si in zip(Zt, y, s):
        t = sum(a * b for a, b in zip(theta, z))
        total += si * ((max(t, 0) + math.log1p(math.exp(-abs(t)))) - yi * t)
    return total


def _grad_hess(theta, Zt, y, s, prec):
    d = len(theta)
    g = [prec * t for t in theta]
    H = [[prec if i == j else 0.0 for j in range(d)] for i in range(d)]
    for z, yi, si in zip(Zt, y, s):
        p = sigmoid(sum(a * b for a, b in zip(theta, z)))
        err, curv = si * (p - yi), si * p * (1 - p)
        for i in range(d):
            g[i] += err * z[i]
            ci = curv * z[i]
            for j in range(i + 1):
                H[i][j] += ci * z[j]
    for i in range(d):
        for j in range(i):
            H[j][i] = H[i][j]
    return g, H


def fit_posterior(names: list[str], X: list[list[float]], y: list[int], s: list[float],
                  sigma0: float = SIGMA0) -> Posterior:
    """The Laplace belief from rows X (raw inputs), labels y and row weights s. No rows → the prior."""
    d = len(names)
    if not X:
        return Posterior(list(names), [0.0] * (d + 1),
                         [[sigma0 if i == j else 0.0 for j in range(d + 1)] for i in range(d + 1)],
                         [0.0] * d, [1.0] * d, 0.0)
    Z, means, stds = standardise(X)
    Zt = [[1.0] + z for z in Z]
    prec = 1.0 / (sigma0 * sigma0)
    theta = [0.0] * (d + 1)
    for _ in range(NEWTON_STEPS):
        g, H = _grad_hess(theta, Zt, y, s, prec)
        step = _solve_spd(cholesky(H), g)
        # Halve the step while it would make things worse; a full step almost always works.
        base, t = _objective(theta, Zt, y, s, prec), 1.0
        while t > 1e-6:
            trial = [a - t * b for a, b in zip(theta, step)]
            if _objective(trial, Zt, y, s, prec) <= base + 1e-12:
                break
            t /= 2
        theta = trial
        if max(abs(t * b) for b in step) < NEWTON_TOL:
            break
    _, H = _grad_hess(theta, Zt, y, s, prec)
    return Posterior(list(names), theta, cholesky(inverse_spd(H)), means, stds, float(sum(s)))


def posterior_from_rows(names: list[str], rows: list[Row], now: float) -> Posterior:
    """The belief from the discovery rows only, each weighted by its label weight × its fading."""
    disc = discovery_rows(rows)
    return fit_posterior(names, [r.x for r in disc], [r.label for r in disc],
                         [r.weight * fade(r.at, now) for r in disc])


def pick_contextual(pool: list[Suggestion], post: Posterior, rng: random.Random) -> Suggestion | None:
    """One Thompson draw: sample weights once, score every pool card with them, the top score wins.

    Cards are scored at rank 1 and not as the discovery pick (`from_suggestion`), as v2 ranks: position
    shouldn't decide. Ties go to the first card, the ranker's order.
    """
    if not pool:
        return None
    theta = post.sample(rng)
    best, best_score = None, -math.inf
    for s in pool:
        sc = post.score(theta, from_suggestion(s, s.boost / CHANNEL_BOOST))
        if sc > best_score:
            best, best_score = s, sc
    return best
