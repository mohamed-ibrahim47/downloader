"""v3: a small neural network, trained by backpropagation written by hand (design v3, D1–D2).

    z = the 8 inputs, standardised (as v2)
    a = W1 · z + b1           one number per hidden neuron ("pre-activation")
    h = act(a)                the activation bends each one: ReLU by default
    P(download) = sigmoid(w2 · h + b2)

v2 (`learn.py`) is this with no hidden layer. The hidden layer is what lets the model learn interactions:
"long videos are bad, unless they're from a favourite channel".

Training minimises the same loss as v2: weighted log-loss plus an L2 penalty on the weights (not the biases).
Backpropagation is the chain rule run from the output back to the inputs:

    output error   δ = s · (p − y)                         (as in v2)
    ∂L/∂w2[i] = δ · h[i]          ∂L/∂b2 = δ
    hidden error   e[i] = δ · w2[i] · act′(a[i])          the error, passed back through w2 and the bend
    ∂L/∂W1[i][j] = e[i] · z[j]    ∂L/∂b1[i] = e[i]

Pure Python on purpose: the exe has no numpy, and every step stays readable.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass

from .learn import sigmoid, standardise

HIDDEN = 8  # hidden neurons: 8×8 + 8 + 8 + 1 = 81 weights for 8 inputs
ACTIVATION = "relu"
LAMBDA = 0.001  # L2 strength
RATE = 0.05  # learning rate
MOMENTUM = 0.9  # how much of the last step carries into the next
BATCH = 32  # rows per step
MAX_EPOCHS = 200  # passes over the rows, at most
PATIENCE = 15  # stop after this many passes without a better held-back loss
HOLDOUT = 0.15  # the newest share of rows held back to decide when to stop
SEED = 0
MAX_ROWS = 2000  # newest rows only: backprop in pure Python is slow
LEAK = 0.01  # leaky ReLU's slope below zero


def _tanh_slope(a: float) -> float:
    t = math.tanh(a)
    return 1.0 - t * t


def _sigmoid_slope(a: float) -> float:
    p = sigmoid(a)
    return p * (1.0 - p)


# name → (the activation, its slope). The slope is what backprop multiplies the error by.
ACTIVATIONS = {
    "relu": (lambda a: a if a > 0 else 0.0, lambda a: 1.0 if a > 0 else 0.0),
    "leaky_relu": (lambda a: a if a > 0 else LEAK * a, lambda a: 1.0 if a > 0 else LEAK),
    "tanh": (math.tanh, _tanh_slope),
    "sigmoid": (sigmoid, _sigmoid_slope),
}


@dataclass
class Net:
    """A trained v3 model. Weights are in standardised units, like v2's."""
    names: list[str]
    W1: list[list[float]]  # [hidden][input]
    b1: list[float]
    w2: list[float]
    b2: float
    means: list[float]
    stds: list[float]
    activation: str = ACTIVATION
    epochs: int = 0  # passes it was trained for
    trainer: str = "v3"

    def size(self) -> int:
        return sum(map(len, self.W1)) + len(self.b1) + len(self.w2) + 1

    def logit_z(self, z: list[float]) -> float:
        act = ACTIVATIONS[self.activation][0]
        return self.b2 + sum(v * act(b + sum(w * zj for w, zj in zip(row, z)))
                             for row, b, v in zip(self.W1, self.b1, self.w2))

    def logit(self, x: list[float]) -> float:
        return self.logit_z([(xj - m) / s for xj, m, s in zip(x, self.means, self.stds)])

    def prob(self, x: list[float]) -> float:
        return sigmoid(self.logit(x))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Net":
        return cls(**json.loads(text))


def init_net(names: list[str], hidden: int, activation: str, rng: random.Random,
             means: list[float] | None = None, stds: list[float] | None = None) -> Net:
    """Random start. All-zero weights would make every hidden neuron the same forever (symmetry).

    The spread keeps the signal about the same size through the layer: He (√(2/n)) for ReLU, which zeroes
    half its inputs, and Xavier (√(1/n)) for tanh and sigmoid.
    """
    d = len(names)
    spread = math.sqrt((2.0 if "relu" in activation else 1.0) / d)
    W1 = [[rng.gauss(0, spread) for _ in range(d)] for _ in range(hidden)]
    w2 = [rng.gauss(0, math.sqrt(1.0 / hidden)) for _ in range(hidden)]
    return Net(list(names), W1, [0.0] * hidden, w2, 0.0, means or [0.0] * d, stds or [1.0] * d, activation)


def loss_and_grads(net: Net, Z, y, s, lam: float) -> tuple[float, dict]:
    """The loss over standardised rows Z, and its gradient for every weight: backprop, one row at a time."""
    act, slope = ACTIVATIONS[net.activation]
    H, d, n = len(net.W1), len(Z[0]), len(Z)
    gW1 = [[0.0] * d for _ in range(H)]
    gb1, gw2, gb2 = [0.0] * H, [0.0] * H, 0.0
    loss = 0.0
    for z, yi, si in zip(Z, y, s):
        # forward
        a = [b + sum(w * zj for w, zj in zip(row, z)) for row, b in zip(net.W1, net.b1)]
        h = [act(ai) for ai in a]
        t = net.b2 + sum(v * hi for v, hi in zip(net.w2, h))
        loss += si * ((max(t, 0) + math.log1p(math.exp(-abs(t)))) - yi * t)  # log-loss, safe form (v2)
        # backward
        delta = si * (sigmoid(t) - yi)
        gb2 += delta
        for i in range(H):
            gw2[i] += delta * h[i]
            e = delta * net.w2[i] * slope(a[i])
            if e:
                gb1[i] += e
                row = gW1[i]
                for j in range(d):
                    row[j] += e * z[j]
    penalty = sum(w * w for row in net.W1 for w in row) + sum(v * v for v in net.w2)
    grads = {"W1": [[g / n + lam * w for g, w in zip(grow, wrow)] for grow, wrow in zip(gW1, net.W1)],
             "b1": [g / n for g in gb1],
             "w2": [g / n + lam * v for g, v in zip(gw2, net.w2)],
             "b2": gb2 / n}
    return loss / n + lam / 2 * penalty, grads


def _flat(net: Net) -> list[float]:
    return [w for row in net.W1 for w in row] + net.b1 + net.w2 + [net.b2]


def _flat_grads(g: dict) -> list[float]:
    return [w for row in g["W1"] for w in row] + g["b1"] + g["w2"] + [g["b2"]]


def _set_flat(net: Net, p: list[float]) -> None:
    H, d = len(net.W1), len(net.W1[0])
    net.W1 = [p[i * d:(i + 1) * d] for i in range(H)]
    k = H * d
    net.b1, net.w2, net.b2 = p[k:k + H], p[k + H:k + 2 * H], p[k + 2 * H]


def _data_loss(net: Net, Z, y, s) -> float:
    """Weighted log-loss with no penalty: what early stopping watches on the held-back rows."""
    total = 0.0
    for z, yi, si in zip(Z, y, s):
        t = net.logit_z(z)
        total += si * ((max(t, 0) + math.log1p(math.exp(-abs(t)))) - yi * t)
    return total / max(sum(s), 1e-12)


class NetTrainer:
    """Mini-batch gradient descent with momentum, from a seeded random start.

    With `epochs=None` the newest `holdout` share of rows (rows come in time order) is held back, and training
    stops once their loss hasn't improved for `patience` passes; the best pass is kept and its number saved
    as `Net.epochs`. With `epochs=k` it trains on every row for exactly k passes: that is how the final net
    is trained, using the count the held-back run found.
    """
    name = "v3"

    def __init__(self, hidden: int = HIDDEN, activation: str = ACTIVATION, lam: float = LAMBDA, rate: float = RATE,
                 momentum: float = MOMENTUM, batch: int = BATCH, max_epochs: int = MAX_EPOCHS,
                 patience: int = PATIENCE, holdout: float = HOLDOUT, seed: int = SEED, epochs: int | None = None):
        self.hidden, self.activation, self.lam, self.rate = hidden, activation, lam, rate
        self.momentum, self.batch, self.max_epochs, self.patience = momentum, batch, max_epochs, patience
        self.holdout, self.seed, self.epochs = holdout, seed, epochs

    def fit(self, names: list[str], X: list[list[float]], y: list[int], s: list[float]) -> Net:
        X, y, s = X[-MAX_ROWS:], y[-MAX_ROWS:], s[-MAX_ROWS:]
        Z, means, stds = standardise(X)
        rng = random.Random(self.seed)
        net = init_net(names, self.hidden, self.activation, rng, means, stds)
        hold = 0 if self.epochs is not None or len(Z) < 20 else max(1, round(len(Z) * self.holdout))
        cut = len(Z) - hold
        train, check = list(range(cut)), (Z[cut:], y[cut:], s[cut:])
        passes = self.epochs if self.epochs is not None else self.max_epochs

        params = _flat(net)
        velocity = [0.0] * len(params)
        best, best_params, best_epoch, waited = math.inf, params[:], passes, 0
        for epoch in range(1, passes + 1):
            rng.shuffle(train)
            for start in range(0, len(train), self.batch):
                idx = train[start:start + self.batch]
                _, g = loss_and_grads(net, [Z[i] for i in idx], [y[i] for i in idx], [s[i] for i in idx], self.lam)
                # Momentum: the step is mostly the last step, nudged by the new gradient.
                velocity = [self.momentum * v - self.rate * gi for v, gi in zip(velocity, _flat_grads(g))]
                params = [p + v for p, v in zip(params, velocity)]
                _set_flat(net, params)
            if hold:
                held = _data_loss(net, *check)
                if held < best - 1e-9:
                    best, best_params, best_epoch, waited = held, params[:], epoch, 0
                else:
                    waited += 1
                    if waited >= self.patience:
                        break
        if hold:
            _set_flat(net, best_params)
        net.epochs = best_epoch if hold else passes
        return net
