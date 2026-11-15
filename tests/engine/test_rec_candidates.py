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


def test_file_names_never_reach_a_youtube_search():
    """A file name can carry a person's name (a wedding video, a CV in a zip): file downloads shape the
    ranking, but never what is sent to YouTube (final review, finding 1)."""
    from downloader.engine.models import Download, Kind
    from downloader.engine.recommend.profile import build_profile
    now = 1_800_000_000.0
    vids = [Download(url="u", kind=Kind.VIDEO, filename="f", folder="d", title=t, added_at=now - 86400)
            for t in ("piano cover song", "guitar chords lesson", "jazz piano live")]
    files = [Download(url="u", kind=Kind.FILE, filename=n, folder="d", added_at=now)
             for n in ("Ahmed_Hassan_Wedding.mp4", "JohnSmith_Resume.zip", "Blender_3D_Modeling-4.2-x64.msi")]
    p = build_profile(vids + files, [], now)
    assert any(d.title.startswith("ahmed") for d in p.docs)  # still in the profile, for ranking
    sent = " ".join(q for q, _kind in plan(p))
    for word in ("ahmed", "hassan", "wedding", "john", "smith", "resume", "blender", "modeling"):
        assert word not in sent
    newest = [q for q, kind in plan(p) if kind == "newest"]
    video_words = {"piano", "cover", "song", "guitar", "chords", "lesson", "jazz", "live"}
    assert newest and set(newest[0].split()) <= video_words  # the newest *video* instead


def test_only_files_means_no_searches_from_downloads():
    from downloader.engine.models import Download, Kind
    from downloader.engine.recommend.profile import build_profile
    now = 1_800_000_000.0
    files = [Download(url="u", kind=Kind.FILE, filename=n, folder="d", added_at=now)
             for n in ("Blender_3D_Modeling.msi", "krita_painting.exe", "python_course_week1.mp4")]
    assert plan(build_profile(files, [], now)) == []
