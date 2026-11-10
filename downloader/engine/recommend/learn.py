"""Logistic regression, written by hand (design C4).

The model:   P(download | shown) = sigmoid(b + w · z),   z = the inputs, standardised.

Training finds w and b that make the logged outcomes most likely. It minimises the weighted log-loss
plus an L2 penalty that keeps the weights small:

    L = (1/n) · Σ sᵢ · lossᵢ  +  λ/2 · ‖w‖²      lossᵢ = −[yᵢ·log pᵢ + (1 − yᵢ)·log(1 − pᵢ)]

Its gradient is short, which is why logistic regression is the classic first model to train by hand:

    ∂L/∂w = (1/n) · Σ sᵢ · (pᵢ − yᵢ) · zᵢ  +  λ·w        ∂L/∂b = (1/n) · Σ sᵢ · (pᵢ − yᵢ)

Gradient descent repeats `w -= rate · ∂L/∂w` until the loss stops falling.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

LAMBDA = 0.01  # strength of the L2 penalty
RATE = 0.5  # learning rate: how big each step is
EPOCHS = 500  # at most this many passes over the rows
TOLERANCE = 1e-7  # stop when a pass improves the loss by less than this


def sigmoid(t: float) -> float:
    """1 / (1 + e^−t), written so a large |t| can't overflow."""
    if t >= 0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


def standardise(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Each column minus its mean, divided by its spread. A constant column keeps spread 1 (and becomes 0)."""
    n, d = len(X), len(X[0])
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    Z = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X]
    return Z, means, stds


@dataclass
class Weights:
    """A trained model. `w` is in standardised units, so weights can be compared with each other."""
    names: list[str]
    w: list[float]
    b: float
    means: list[float]
    stds: list[float]
    trainer: str

    def logit(self, x: list[float]) -> float:
        return self.b + sum(wj * (xj - mj) / sj for wj, xj, mj, sj in zip(self.w, x, self.means, self.stds))

    def prob(self, x: list[float]) -> float:
        return sigmoid(self.logit(x))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Weights":
        return cls(**json.loads(text))


def loss_and_grad(Z, y, s, w, b, lam):
    """The loss L and its gradient (∂L/∂w, ∂L/∂b) at (w, b), over standardised rows Z."""
    n, d = len(Z), len(w)
    loss, gw, gb = 0.0, [0.0] * d, 0.0
    for z, yi, si in zip(Z, y, s):
        t = b + sum(wj * zj for wj, zj in zip(w, z))
        p = sigmoid(t)
        # log(1 + e^t) − y·t is the log-loss written without log(0) when p rounds to 0 or 1.
        loss += si * ((max(t, 0) + math.log1p(math.exp(-abs(t)))) - yi * t)
        err = si * (p - yi)
        gb += err
        for j in range(d):
            gw[j] += err * z[j]
    loss = loss / n + lam / 2 * sum(wj * wj for wj in w)
    gw = [g / n + lam * wj for g, wj in zip(gw, w)]
    return loss, gw, gb / n


class HandTrainer:
    """Full-batch gradient descent, from all-zero weights, so the same rows always give the same model."""
    name = "hand"

    def __init__(self, lam: float = LAMBDA, rate: float = RATE, epochs: int = EPOCHS,
                 tolerance: float = TOLERANCE):
        self.lam, self.rate, self.epochs, self.tolerance = lam, rate, epochs, tolerance

    def fit(self, names: list[str], X: list[list[float]], y: list[int], s: list[float]) -> Weights:
        Z, means, stds = standardise(X)
        w, b = [0.0] * len(names), 0.0
        last = math.inf
        for _ in range(self.epochs):
            loss, gw, gb = loss_and_grad(Z, y, s, w, b, self.lam)
            if last - loss < self.tolerance:
                break
            last = loss
            w = [wj - self.rate * g for wj, g in zip(w, gw)]
            b -= self.rate * gb
        return Weights(list(names), w, b, means, stds, self.name)
