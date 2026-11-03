import math

import pytest

from downloader.engine.recommend.tfidf import cosine, fit_idf, vector, weighted_sum

# The spec's worked example (A5): a raw split, with no stop words, so the numbers can be checked by hand.
DOCS = {"A": "taylor swift blank space", "B": "taylor swift style live", "C": "taylor swift shake it off",
        "D": "blank space piano cover", "E": "python tutorial for beginners"}


def test_worked_example_matches_the_spec():
    toks = {k: v.split() for k, v in DOCS.items()}
    idf = fit_idf(list(toks.values()))
    assert idf["taylor"] == pytest.approx(math.log(5 / 3)) and idf["blank"] == pytest.approx(math.log(5 / 2))
    profile = weighted_sum([(vector(toks["A"], idf), 1.0), (vector(toks["B"], idf), 1.0)])
    assert profile["style"] == pytest.approx(0.402, abs=1e-3)
    scores = {k: cosine(profile, vector(toks[k], idf)) for k in "CDE"}
    assert scores["D"] == pytest.approx(0.214, abs=1e-3)
    assert scores["C"] == pytest.approx(0.121, abs=1e-3)
    assert scores["E"] == 0.0


def test_empty_inputs_are_zero_not_errors():
    assert vector([], {"a": 1.0}) == {}
    assert cosine({}, {"a": 1.0}) == 0.0 and cosine({"a": 0.0}, {"a": 1.0}) == 0.0
    assert fit_idf([]) == {}


def test_negative_weight_pulls_a_word_down():
    s = weighted_sum([({"x": 1.0, "y": 1.0}, 1.0), ({"y": 1.0}, -0.5)])
    assert s == {"x": 1.0, "y": 0.5}
