import math

import pytest

from downloader.engine.recommend.embed import Vectors
from downloader.engine.recommend.meaning import MEANING_TOP, MeaningIndex, v4_score
from downloader.engine.recommend.profile import Profile, ProfileDoc
from downloader.engine.recommend.rank import CHANNEL_BOOST

# Three tastes on three axes; "guitar" sits with music, "travel" in between everything.
VEC = Vectors(["piano", "lion", "python", "guitar", "travel"],
              [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.1, 0], [0.6, 0.6, 0.5]], [1] * 5)


def _doc(word, weight=1.0):
    return ProfileDoc(None, word, [word], weight)


def _profile(*docs):
    return Profile(list(docs), {})


def test_best_three_not_average():
    p = _profile(*[_doc(w) for w in ("piano", "lion", "python") for _ in range(3)])
    idx = MeaningIndex(p, VEC, {})
    music = idx.score(["guitar"])
    average = 1 / math.sqrt(3) * (1 + 0.1) / math.sqrt(1 + 0.01)  # cosine with one averaged profile vector
    assert MEANING_TOP == 3
    assert music > 0.95 > average
    assert music > idx.score(["travel"])


def test_fewer_than_three_uses_what_there_is():
    idx = MeaningIndex(_profile(_doc("piano")), VEC, {})
    assert idx.score(["piano"]) == pytest.approx(1.0)


def test_recency_weight_scales_match():
    full = MeaningIndex(_profile(_doc("piano", 1.0)), VEC, {}).score(["piano"])
    half = MeaningIndex(_profile(_doc("piano", 0.5)), VEC, {}).score(["piano"])
    assert half == pytest.approx(full / 2)


def test_dismissed_close_title_pulls_down():
    liked = [_doc("piano"), _doc("piano"), _doc("piano")]
    before = MeaningIndex(_profile(*liked), VEC, {}).score(["guitar"])
    after = MeaningIndex(_profile(*liked, _doc("guitar", -0.5)), VEC, {}).score(["guitar"])
    assert after == pytest.approx(before - 0.5, abs=1e-6)


def test_meaning_zero_for_unknown_title():
    idx = MeaningIndex(_profile(_doc("piano")), VEC, {})
    assert idx.score(["عمرو", "🔥"]) == 0.0 and idx.score([]) == 0.0
    assert MeaningIndex(_profile(_doc("unknown")), VEC, {}).score(["piano"]) == 0.0


def test_v4_score_equals_v1_at_mix_zero():
    assert v4_score(0.3, 0.9, 0.5, 0.0) == pytest.approx(0.3 + CHANNEL_BOOST * 0.5)
    assert v4_score(0.3, 0.9, 0.5, 0.5) == pytest.approx(0.6 + CHANNEL_BOOST * 0.5)
