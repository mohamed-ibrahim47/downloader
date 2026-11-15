import pytest

from downloader.engine.recommend import starter
from downloader.engine.recommend.features import NAMES


def test_the_starter_loads_and_scores():
    assert starter.RANKER in (None, "v2", "v3")
    m = starter.model()
    if starter.RANKER is None:
        assert m is None
        return
    assert m.names == NAMES
    p = m.prob([0.5, 0.2, 2, 2.0, 1, 0, 0.0, 0])
    assert 0.0 < p < 1.0


def test_the_starter_was_checked_across_months_and_beat_v1():
    if starter.RANKER is None:
        return
    check = starter.CHECK
    assert check["v1"] < check[starter.RANKER]  # used only when it beats v1 on the other month
    assert starter.ROWS >= 200 and starter.DOWNLOADS >= 15


@pytest.mark.real_starter
def test_the_shipped_starter_vectors_load_and_make_sense():
    from downloader.engine.recommend import starter_vectors
    from downloader.engine.recommend.embed import cosine
    v = starter_vectors.vectors()
    assert v is not None and v.dim == 32 and len(v) == starter_vectors.TRAINED_ON["words"] > 1000
    assert len(v.counts) == len(v)
    a, b = starter_vectors.EXAMPLE
    assert cosine(v.get(a), v.get(b)) > 0.5  # the banner's example is true of these vectors
    assert starter_vectors.DEFAULT_ON == (starter_vectors.CHECK["v4"] > starter_vectors.CHECK["v1"])
    assert 0 < starter_vectors.MEANING_MIX < 1
