import pytest

from downloader.engine.errors import DownloadError
from downloader.engine.recommend.candidates import Candidate, fetch, keep, plan, queries
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


def test_the_newest_download_always_gets_a_search():
    # Old favourites outscore it (shorter titles, more of them), but a new interest must reach YouTube.
    old = [ProfileDoc(str(i), t, [], 0.7) for i, t in enumerate(["Blank Space", "Style Era", "Shake Cover"])]
    new = ProfileDoc("9", "Interstellar Official Trailer Christopher Nolan Movie Scene", [], 1.0)
    p = Profile(docs=[new, *old], channels={"TS": 1.0, "Other": 0.5})
    q = queries(p)
    assert q[:2] == ["TS", "Other"] and "interstellar trailer" in q and len(q) == 4


def test_a_one_word_newest_title_is_still_searched():
    p = Profile(docs=[ProfileDoc("9", "Interstellar", [], 1.0), ProfileDoc("1", "Blank Space", [], 0.5)],
                channels={})
    assert queries(p)[0] == "interstellar"


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


def test_plan_names_the_kind_of_each_search():
    # The learning log records how a video was found, so v2 can learn which kind of search suits the owner.
    old = [ProfileDoc(str(i), t, [], 0.7) for i, t in enumerate(["Blank Space", "Style Era", "Shake Cover"])]
    new = ProfileDoc("9", "Interstellar Official Trailer Christopher Nolan Movie Scene", [], 1.0)
    p = Profile(docs=[new, *old], channels={"TS": 1.0, "Other": 0.5})
    kinds = dict(plan(p))
    assert [q for q, _ in plan(p)] == queries(p)
    assert kinds["TS"] == kinds["Other"] == "channel" and kinds["interstellar trailer"] == "newest"
    assert sorted(kinds.values()).count("words") == 1


def test_fetch_tags_each_candidate_with_its_search_kind():
    hit = SearchHit("v1", "u", "t", "c", 200)
    got = fetch([("TS", "channel"), ("blank space", "words")], lambda q, limit: [hit])
    assert [c.kind for c in got] == ["channel", "words"]
    assert fetch(["plain"], lambda q, limit: [hit])[0].kind is None  # a bare query has no kind
