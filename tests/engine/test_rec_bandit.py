import random

import pytest

from downloader.engine.recommend.bandit import (ARMS, Arm, arm_stats, discovery_rows, fade, kind_of, kind_of_row,
                                                pick_source, propensity)
from downloader.engine.recommend.candidates import Candidate
from downloader.engine.recommend.dataset import Row
from downloader.engine.recommend.features import _vector
from downloader.engine.recommend.rank import Suggestion

NOW = 1_800_000_000.0


def row(kind="words", label=0, weight=1.0, at=NOW, discovery=True):
    x = _vector(0.1, 0.0, 0, 200, kind, 5 if discovery else 1, discovery)
    return Row("v", x, label, weight, 1.0, at)


def sugg(vid, kind="words"):
    return Suggestion(Candidate(vid, "u", vid, None, 200, "q", kind), 0.1, 0.1, 0.0, None, [])


def test_arms_are_the_three_search_kinds():
    assert ARMS == ("channel", "newest", "words")


def test_arm_stats_counts_yes_no_and_ignored_weights():
    rows = [row(kind="newest", label=1, weight=1.0), row(kind="newest", label=0, weight=1.0),
            row(kind="newest", label=0, weight=0.3), row(kind="words", label=1, weight=1.0, discovery=False)]
    arms = arm_stats(rows, now=NOW)
    assert arms["newest"].yes == pytest.approx(1.0) and arms["newest"].no == pytest.approx(1.3)
    assert arms["words"].yes == 0 and arms["channel"].a == 1 and arms["channel"].b == 1  # non-discovery row ignored


def test_arm_rate_is_the_beta_mean():
    assert Arm("x", 3, 1).rate == pytest.approx(4 / 6)


def test_rows_fade_with_a_30_day_half_life():
    arms = arm_stats([row(kind="channel", label=1, at=NOW - 30 * 86400)], now=NOW)
    assert arms["channel"].yes == pytest.approx(0.5)
    assert fade(NOW + 5, NOW) == 1.0  # a row "from the future" (clock skew) never counts more than once


def test_kind_of_row_reads_the_kind_inputs():
    assert [kind_of_row(row(kind=k).x) for k in ("channel", "newest", "words", None)] == \
        ["channel", "newest", "words", "words"]


def test_discovery_rows_keeps_only_discovery_cards():
    assert len(discovery_rows([row(), row(discovery=False)])) == 1


def test_none_kind_counts_as_words():
    assert kind_of(sugg("a", kind=None)) == "words"


def test_better_arm_wins_most_draws():
    arms = {"channel": Arm("channel", 20, 2), "newest": Arm("newest", 1, 20), "words": Arm("words", 1, 20)}
    pool = [sugg("n", "newest"), sugg("c", "channel"), sugg("w", "words")]
    wins = sum(pick_source(pool, arms, random.Random(i))[0].candidate.video_id == "c" for i in range(200))
    assert wins > 180


def test_arm_pick_is_its_highest_ranked_pool_video():  # pool is in ranker order
    pool = [sugg("c1", "channel"), sugg("c2", "channel")]
    assert pick_source(pool, arm_stats([], NOW), random.Random(1))[0].candidate.video_id == "c1"


def test_only_arms_in_pool_can_win():
    s, draws = pick_source([sugg("w", "words")], {**arm_stats([], NOW), "channel": Arm("channel", 99, 0)},
                           random.Random(0))
    assert s.candidate.video_id == "w" and set(draws) == {"words"}


def test_empty_pool_gives_none():
    assert pick_source([], arm_stats([], NOW), random.Random(0)) is None


def test_propensity_of_a_sure_pick_is_one_and_an_even_one_is_about_half():
    sure = {"channel": Arm("channel", 500, 0), "newest": Arm("newest", 0, 500), "words": Arm("words", 0, 0)}
    pool = [sugg("c", "channel"), sugg("n", "newest")]
    fn = lambda p, r: pick_source(p, sure, r)  # noqa: E731
    assert propensity(fn, pool, "c", random.Random(0)) == pytest.approx(1.0)
    even = arm_stats([], NOW)
    assert 0.35 < propensity(lambda p, r: pick_source(p, even, r), pool, "c", random.Random(0)) < 0.65


def test_propensity_does_not_shift_the_callers_rng_by_its_draws():
    rng = random.Random(5)
    propensity(lambda p, r: pick_source(p, arm_stats([], NOW), r), [sugg("c", "channel")], "c", rng)
    ref = random.Random(5)
    ref.random()
    assert rng.random() == ref.random()


def test_propensity_is_never_zero_for_the_card_really_picked():
    # A pick with a tiny chance can miss in all 200 repeats; the real draw counts as one, so it is > 0
    # (off-policy evaluation divides by it later).
    never = lambda p, r: (p[1], {})  # noqa: E731 — the repeats always pick the other card
    pool = [sugg("c", "channel"), sugg("n", "newest")]
    got = propensity(never, pool, "c", random.Random(0))
    assert 0 < got <= 1 / 200
