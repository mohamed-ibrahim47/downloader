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
