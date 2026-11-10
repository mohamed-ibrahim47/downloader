import math

import pytest

from downloader.engine.recommend.candidates import Candidate
from downloader.engine.recommend.features import NAMES, from_logged, from_suggestion
from downloader.engine.recommend.rank import Suggestion


def _logged(**over):
    f = {"similarity": 0.4, "boost": 0.075, "affinity": 0.5, "duration": 600, "rank": 3, "discovery": False,
         "shared": 2, "kind": "channel"}
    f.update(over)
    return f


def test_names_are_the_designs_eight_inputs():
    assert NAMES == ["similarity", "affinity", "shared", "log_duration", "kind_channel", "kind_newest",
                     "log_rank", "discovery"]


def test_a_logged_row_becomes_numbers_in_name_order():
    x = from_logged(_logged())
    assert x == pytest.approx([0.4, 0.5, 2, math.log(1 + 10), 1, 0, math.log(3), 0])


def test_kind_is_one_hot_with_words_as_the_base():
    assert from_logged(_logged(kind="newest"))[4:6] == [0, 1]
    assert from_logged(_logged(kind="words"))[4:6] == [0, 0]
    assert from_logged(_logged(kind=None))[4:6] == [0, 0]


def test_missing_duration_counts_as_zero_and_discovery_as_one():
    x = from_logged(_logged(duration=None, discovery=True))
    assert x[3] == 0 and x[7] == 1


def test_a_suggestion_is_scored_at_rank_one_and_not_as_discovery():
    # Position bias: rank and discovery are inputs in training, but every candidate is scored at the
    # same position, so those two terms can't change the order.
    c = Candidate("v", "u", "t", "Ch", 600, "q", "newest")
    s = Suggestion(c, 0.5, 0.4, 0.1, None, ["a", "b"], discovery=True)
    x = from_suggestion(s, affinity=0.5)
    assert x == pytest.approx([0.4, 0.5, 2, math.log(11), 0, 1, 0, 0])


def test_at_top_matches_how_a_suggestion_is_scored():
    from downloader.engine.recommend.features import at_top
    assert at_top(from_logged(_logged(rank=7, discovery=True)))[6:] == [0, 0]
