import pytest

from downloader.engine.errors import DownloadError
from downloader.engine.recommend.candidates import Candidate, fetch, keep, queries
from downloader.engine.recommend.profile import Profile, ProfileDoc
from downloader.engine.youtube import SearchHit


def test_queries_two_channels_then_two_word_pairs():
    p = Profile(docs=[ProfileDoc("1", "Blank Space", ["blank", "space", "ch:ts"], 1.0),
                      ProfileDoc("2", "Style Era", ["style", "era", "ch:ts"], 1.0),
                      ProfileDoc("3", "Piano Cover", ["piano", "cover", "ch:other"], 0.5)],
                channels={"TS": 1.0, "Other": 0.5, "Third": 0.2})
    q = queries(p)
    assert q[:2] == ["TS", "Other"] and len(q) == 4
    assert set(q[2:]) <= {"blank space", "style era", "piano cover"}


def test_queries_skip_negative_docs_and_duplicates():
    p = Profile(docs=[ProfileDoc("1", "Blank Space", ["blank", "space"], 1.0),
                      ProfileDoc("2", "Blank Space", ["blank", "space"], 1.0),
                      ProfileDoc("x", "Bad Song", ["bad", "song"], -0.5)], channels={})
    assert queries(p) == ["blank space"]


def test_pairs_come_from_the_title_not_the_tags():
    # Tags are often SEO filler; spec B5 builds pairs from words in one download's title.
    p = Profile(docs=[ProfileDoc("1", "Song Title", ["song", "title", "trending", "viral", "viral"], 1.0)],
                channels={})
    assert queries(p) == ["song title"]


def test_blocked_channels_are_not_queries():
    p = Profile(docs=[], channels={"TS": 1.0, "Other": 0.5, "Third": 0.2})
    assert queries(p, blocked={"TS"}) == ["Other", "Third"]


def test_fetch_keeps_successes_when_a_query_fails():
    def search(q, limit):
        if q == "bad":
            raise RuntimeError("offline")
        return [SearchHit(q + "1", "u", "t", None, 100)]
    assert [c.video_id for c in fetch(["good", "bad"], search)] == ["good1"]


def test_fetch_raises_when_every_query_fails():
    with pytest.raises(DownloadError):
        fetch(["a", "b"], lambda q, limit: (_ for _ in ()).throw(RuntimeError("offline")))


def test_keep_filters_and_dedupes():
    c = lambda vid, ch="C", dur=200: Candidate(vid, "u", "t", ch, dur, "q")
    cands = [c("ok"), c("ok"), c("have"), c("nope"), c("blocked", ch="Bad"), c("short", dur=59),
             c("live", dur=None), c("edge", dur=60)]
    assert [x.video_id for x in keep(cands, {"have"}, {"nope"}, {"Bad"})] == ["ok", "edge"]
