"""Our own word embeddings: word2vec skip-gram with negative sampling, by hand (design v4, E2).

Every word gets a short list of numbers (a vector). Training reads titles and, for each word and a word
near it, pushes their vectors to agree; for a few random words, it pushes them apart. After many titles,
words that sit among the same neighbours ("piano", "guitar") point the same way, with nobody telling it so.

Two tables are learned: `inp` (the word as the center) and `out` (the word as a neighbour). `inp` is kept;
`out` is a training aid and thrown away. Plain Python: the exe has no numpy.
"""
from __future__ import annotations

import base64
import math
import random
import struct
from collections import Counter

DIM = 32
WINDOW = 3  # words each side; titles are short
NEGATIVES = 5  # random "not neighbours" per real pair
SUBSAMPLE = 1e-4  # very frequent words are skipped now and then, so "official" and "video" don't dominate
LR = 0.025
LR_MIN = 0.0001
STARTER_EPOCHS = 5
RETRAIN_EPOCHS = 3
STARTER_MIN_COUNT = 5
USER_MIN_COUNT = 2
MAX_WORDS = 20_000
_TABLE_SIZE = 100_000  # the negative-sampling table


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _sigmoid(x: float) -> float:
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class Vectors:
    """Words and their vectors, in a fixed order (most frequent first)."""

    def __init__(self, words: list[str], vecs: list[list[float]], counts: list[int]):
        self.words = list(words)
        self.vecs = [list(v) for v in vecs]
        self.counts = list(counts)
        self.dim = len(self.vecs[0]) if self.vecs else 0
        self._index = {w: i for i, w in enumerate(self.words)}
        self._unit: list[list[float]] | None = None

    def __len__(self) -> int:
        return len(self.words)

    def get(self, word: str) -> list[float] | None:
        i = self._index.get(word)
        return self.vecs[i] if i is not None else None

    def title_vector(self, tokens: list[str], idf: dict[str, float]) -> list[float] | None:
        """The IDF-weighted mean of the known words' vectors, scaled to length 1. None: no known word."""
        total = [0.0] * self.dim
        found = False
        for t in tokens:
            v = self.get(t)
            if v is None:
                continue
            w = idf.get(t, 1.0)
            for k in range(self.dim):
                total[k] += w * v[k]
            found = True
        if not found:
            return None
        n = math.sqrt(sum(x * x for x in total))
        return [x / n for x in total] if n > 0 else None

    def nearest(self, word: str, n: int = 4) -> list[str]:
        """The `n` words whose vectors point most the same way as `word`'s."""
        i = self._index.get(word)
        if i is None:
            return []
        if self._unit is None:
            self._unit = []
            for v in self.vecs:
                length = math.sqrt(sum(x * x for x in v)) or 1.0
                self._unit.append([x / length for x in v])
        me = self._unit[i]
        scored = [(sum(a * b for a, b in zip(me, u)), w) for j, (u, w) in enumerate(zip(self._unit, self.words))
                  if j != i]
        scored.sort(key=lambda sw: (-sw[0], sw[1]))
        return [w for _, w in scored[:n]]

    # ---- storage: SQLite rows (float32) and the bundled starter (half precision, base64) ----

    def to_rows(self) -> list[tuple[str, bytes, int]]:
        fmt = f"<{self.dim}f"
        return [(w, struct.pack(fmt, *v), c) for w, v, c in zip(self.words, self.vecs, self.counts)]

    @classmethod
    def from_rows(cls, rows, dim: int) -> "Vectors":
        fmt = f"<{dim}f"
        rows = list(rows)
        return cls([r[0] for r in rows], [list(struct.unpack(fmt, r[1])) for r in rows], [r[2] for r in rows])

    def to_b64(self) -> str:
        flat = [x for v in self.vecs for x in v]
        return base64.b64encode(struct.pack(f"<{len(flat)}e", *flat)).decode("ascii")

    @classmethod
    def from_b64(cls, words: list[str], counts: list[int], b64: str, dim: int) -> "Vectors":
        raw = base64.b64decode(b64)
        flat = struct.unpack(f"<{len(raw) // 2}e", raw)
        return cls(words, [list(flat[i * dim:(i + 1) * dim]) for i in range(len(words))], counts)


def _step(inp: list[list[float]], out: list[list[float]], c: int, pairs: list[tuple[int, int]], lr: float) -> None:
    """One center word `c` against its pairs: (word, 1) for a real neighbour, (word, 0) for a random one.

    For each pair the error is (label − sigmoid(inp[c] · out[o])). `out[o]` moves along `inp[c]` by it, and
    the center's change is collected and applied once at the end (the word2vec order).
    """
    center = inp[c]
    dim = len(center)
    change = [0.0] * dim
    for o, label in pairs:
        ctx = out[o]
        g = lr * (label - _sigmoid(sum(center[k] * ctx[k] for k in range(dim))))
        for k in range(dim):
            change[k] += g * ctx[k]
            ctx[k] += g * center[k]
    for k in range(dim):
        center[k] += change[k]


def _vocabulary(titles: list[list[str]], start: Vectors | None, min_count: int,
                max_words: int) -> tuple[list[str], list[int]]:
    counts = Counter(w for t in titles for w in t)
    words: list[str] = []
    totals: list[int] = []
    if start is not None:  # old words stay, even when rare in the new titles
        for w, c in zip(start.words, start.counts):
            words.append(w)
            totals.append(c + counts.get(w, 0))
    have = set(words)
    fresh = sorted((w for w, c in counts.items() if c >= min_count and w not in have), key=lambda w: (-counts[w], w))
    for w in fresh[:max(0, max_words - len(words))]:
        words.append(w)
        totals.append(counts[w])
    return words, totals


def train(titles: list[list[str]], start: Vectors | None = None, dim: int = DIM, window: int = WINDOW,
          negatives: int = NEGATIVES, epochs: int = STARTER_EPOCHS, min_count: int = STARTER_MIN_COUNT,
          max_words: int = MAX_WORDS, seed: int = 0, lr: float = LR, subsample: float = SUBSAMPLE) -> Vectors:
    """Learn vectors from `titles` (each a list of words). With `start`, begin from its vectors and add the
    new words that appear at least `min_count` times; old words keep theirs as a starting point."""
    rng = random.Random(seed)
    if start is not None:
        dim = start.dim
    words, counts = _vocabulary(titles, start, min_count, max_words)
    if not words:
        return Vectors([], [], [])
    index = {w: i for i, w in enumerate(words)}
    inp = [list(start.vecs[i]) for i in range(len(start))] if start is not None else []
    for _ in range(len(words) - len(inp)):
        inp.append([rng.uniform(-0.5, 0.5) / dim for _ in range(dim)])
    out = [[0.0] * dim for _ in words]

    # Negatives are drawn in proportion to count^0.75: common words more often, but less than their share.
    powered = [c ** 0.75 for c in counts]
    total_p = sum(powered)
    table: list[int] = []
    for i, p in enumerate(powered):
        table.extend([i] * max(1, round(p / total_p * _TABLE_SIZE)))

    seqs = [[index[w] for w in t if w in index] for t in titles]
    n_tokens = sum(len(s) for s in seqs) or 1
    keep = [min(1.0, math.sqrt(subsample / (c / n_tokens)) + subsample / (c / n_tokens)) for c in counts]

    steps_total = max(1, epochs * n_tokens)
    done = 0
    for _ in range(epochs):
        order = list(range(len(seqs)))
        rng.shuffle(order)
        for si in order:
            seq = [w for w in seqs[si] if rng.random() < keep[w]]
            for i, c in enumerate(seq):
                rate = max(LR_MIN, lr * (1 - done / steps_total))
                done += 1
                for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                    if j == i:
                        continue
                    o = seq[j]
                    pairs = [(o, 1)]
                    for _ in range(negatives):
                        n = table[rng.randrange(len(table))]
                        if n != o:
                            pairs.append((n, 0))
                    _step(inp, out, c, pairs, rate)
            done += len(seqs[si]) - len(seq)  # skipped words still count toward the learning-rate schedule
    return Vectors(words, inp, counts)
