import random

import pytest

from downloader.engine.models import Download, Kind
from downloader.engine.recommend.candidates import Candidate
from downloader.engine.recommend.profile import build_profile
from downloader.engine.recommend.rank import profile_words, rank

NOW = 1_800_000_000.0


def _profile(*items):
    """Docs from (title, channel) pairs at weight 1.0, with channels worked out as build_profile does."""
    downloads = [Download(url="u", kind=Kind.VIDEO, filename="f", folder="d", title=title, channel=channel,
                          added_at=NOW) for title, channel in items]
    return build_profile(downloads, [], NOW)


def _c(vid, title, channel):
    return Candidate(vid, "u", title, channel, 200, "q")


def test_closer_title_ranks_higher_and_explains_itself():
    p = _profile(("Blank Space", "TS"), ("Style", "TS"), ("Shake It Off", "TS"))
    s = rank(p, [_c("p", "Blank Space Piano Cover", "Pianist"), _c("e", "Python Tutorial", "Code")])
    assert [x.candidate.video_id for x in s] == ["p", "e"]
    assert s[0].because == "Blank Space" and s[0].shared == ["blank", "space"]
    assert s[1].similarity == 0.0 and s[1].because is None


def test_channel_boost_is_015_times_affinity():
    p = _profile(("a", "TS"), ("b", "TS"), ("c", "Other"))
    s = rank(p, [_c("x", "zzz", "Other")])[0]
    assert s.boost == pytest.approx(0.15 * 0.5) and s.score == pytest.approx(s.similarity + s.boost)


def test_at_most_two_per_channel():
    p = _profile(("song one", "TS"), ("song two", "TS"), ("song three", "TS"))
    s = rank(p, [_c(str(i), f"song {i}", "TS") for i in range(5)] + [_c("o", "song other", "Else")])
    assert sum(x.candidate.channel == "TS" for x in s) == 2


def test_missing_channels_are_not_one_channel():
    p = _profile(("song one", None), ("song two", None), ("song three", None))
    assert len(rank(p, [_c(str(i), f"song {i}", None) for i in range(5)])) == 5


def test_discovery_slot_at_five_from_ranks_11_to_30():
    p = _profile(("song", None), ("song", None), ("song", None))
    # 40 channels; equal scores, so the sorted order is simply 00, 01, …, 39
    cands = [_c(f"{i:02}", f"song w{i:02}x", f"ch{i}") for i in range(40)]
    s = rank(p, cands, rng=random.Random(1))
    ids = [x.candidate.video_id for x in s]
    assert len(s) == 10 and len(set(ids)) == 10  # moved into slot 5, never shown twice
    assert s[4].discovery and s[4].because == "Something a little different"
    assert s[4].candidate.video_id in {f"{i:02}" for i in range(10, 30)}


def test_discovery_keeps_a_given_pick_while_it_is_in_range():
    p = _profile(("song", None), ("song", None), ("song", None))
    cands = [_c(f"{i:02}", f"song w{i:02}x", f"ch{i}") for i in range(40)]
    for seed in range(5):
        s = rank(p, cands, rng=random.Random(seed), discovery_id="17")
        assert s[4].candidate.video_id == "17" and s[4].discovery


def test_no_discovery_when_fewer_than_eleven():
    p = _profile(("a", None), ("b", None), ("c", None))
    assert not any(x.discovery for x in rank(p, [_c(str(i), "a", f"c{i}") for i in range(8)]))


def test_noise_only_candidate_scores_zero():
    p = _profile(("a", None), ("b", None), ("c", None))
    assert rank(p, [_c("n", "Official Video 4K", None)])[0].similarity == 0.0


def test_profile_words_skip_channels_highest_first():
    p = _profile(("Blank Space", "TS"), ("Blank Style", "TS"), ("Piano", "TS"))
    words = profile_words(p, [_c("x", "Python", "Code")])
    # piano: tf 1/2 × idf log(4/1) = 0.69 beats blank: 2 × 1/3 × log(4/2) = 0.46. IDF makes the shared word cheap.
    assert words[0][0] == "piano" and words[0][1] == pytest.approx(0.693, abs=1e-3)
    assert not any(w.startswith("ch:") for w, _ in words)
    assert [v for _, v in words] == sorted((v for _, v in words), reverse=True)
