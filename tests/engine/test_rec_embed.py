import random

import pytest

from downloader.engine.recommend import embed
from downloader.engine.recommend.embed import Vectors, cosine, train


def _corpus(n=300):
    """Two topics that never share a word."""
    rng = random.Random(1)
    a, b = ["cat", "dog", "puppy", "kitten", "pet"], ["piano", "guitar", "chords", "song", "cover"]
    return [rng.sample(a, 3) for _ in range(n)] + [rng.sample(b, 3) for _ in range(n)]


def test_same_topic_words_end_closer():
    v = train(_corpus(), dim=16, epochs=5, min_count=2, seed=0, subsample=1.0)  # 10 words: all "frequent"
    assert cosine(v.get("cat"), v.get("dog")) > cosine(v.get("cat"), v.get("piano")) + 0.3
    assert set(v.nearest("piano", 3)) <= {"guitar", "chords", "song", "cover"}


def test_repeatable_with_seed():
    a = train(_corpus(60), dim=8, epochs=2, min_count=2, seed=3)
    b = train(_corpus(60), dim=8, epochs=2, min_count=2, seed=3)
    assert a.to_rows() == b.to_rows()


def test_one_step_moves_pair_together():
    rng = random.Random(0)
    inp = [[rng.uniform(-0.5, 0.5) for _ in range(4)] for _ in range(3)]
    out = [[rng.uniform(-0.5, 0.5) for _ in range(4)] for _ in range(3)]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    before_pos, before_neg = dot(inp[0], out[1]), dot(inp[0], out[2])
    embed._step(inp, out, 0, [(1, 1), (2, 0)], 0.1)
    assert dot(inp[0], out[1]) > before_pos
    assert dot(inp[0], out[2]) < before_neg


def test_min_count_and_max_words():
    titles = [["common", "often", "rare"]] + [["common", "often", "sometimes"]] * 5 + [["common"]] * 5
    v = train(titles, dim=4, epochs=1, min_count=2, seed=0)
    assert v.get("rare") is None and v.get("common") is not None
    v = train(titles, dim=4, epochs=1, min_count=1, max_words=2, seed=0)
    assert v.words == ["common", "often"]


def test_title_vector_unit_and_unknown():
    v = train(_corpus(40), dim=8, epochs=1, min_count=2, seed=0)
    t = v.title_vector(["cat", "dog", "unknownword"], {"cat": 1.0, "dog": 2.0})
    assert sum(x * x for x in t) == pytest.approx(1.0, abs=1e-6)
    assert v.title_vector(["nothing", "known"], {}) is None
    assert v.title_vector([], {}) is None


def test_rows_and_b64_round_trip():
    v = train(_corpus(40), dim=8, epochs=1, min_count=2, seed=0)
    w = Vectors.from_rows(v.to_rows(), 8)
    assert w.to_rows() == v.to_rows()
    h = Vectors.from_b64(v.words, v.counts, v.to_b64(), 8)
    assert h.words == v.words
    for word in v.words:
        assert h.get(word) == pytest.approx(v.get(word), abs=1e-3)


def test_start_keeps_old_words_and_adds_new():
    old = train(_corpus(40), dim=8, epochs=1, min_count=2, seed=0)
    new = train([["violin", "cello", "piano"]] * 3, start=old, dim=8, epochs=1, min_count=2, seed=0)
    assert set(old.words) <= set(new.words) and {"violin", "cello"} <= set(new.words)
    assert new.dim == 8
