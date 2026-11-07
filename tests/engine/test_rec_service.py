import json
import random
import time
from types import SimpleNamespace

import pytest

from downloader.engine.errors import DownloadError
from downloader.engine.models import Download, Kind
from downloader.engine.recommend.service import Recommender
from downloader.engine.store import Store
from downloader.engine.youtube import SearchHit


def _fake_search(q, limit, calls=None):
    """20 distinct hits per query, ids made from the query, 5 channels."""
    if calls is not None:
        calls.append(q)
    key = q.replace(" ", "_")
    return [SearchHit(f"{key}{i:02}", f"https://www.youtube.com/watch?v={key}{i:02}", f"{q} song {i}",
                      f"{q} ch{i % 5}", 200) for i in range(20)]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    for title in ("Blank Space", "Style", "Shake It Off"):
        s.add(Download(url="u", kind=Kind.VIDEO, filename="f", folder="d", title=title, channel="TS"))
    yield s
    s.close()


@pytest.fixture
def clock():
    return SimpleNamespace(now=time.time())


@pytest.fixture
def rec(store, clock):
    return Recommender(store, search=_fake_search, clock=lambda: clock.now, rng=random.Random(0))


def test_refresh_saves_candidates_and_suggestions_log_shown(rec, clock):
    rec.refresh()
    s = rec.suggestions()
    assert s and rec.log.last_refresh() == clock.now
    shown = rec.store.query("SELECT video_id, rank, features FROM rec_events WHERE event = 'shown' ORDER BY rank")
    assert [r[0] for r in shown] == [x.candidate.video_id for x in s] and shown[0][1] == 1
    f = json.loads(shown[0][2])
    assert set(f) == {"similarity", "boost", "affinity", "duration", "rank", "discovery", "shared"}
    assert f["similarity"] == pytest.approx(s[0].similarity) and f["rank"] == 1


def test_needs_refresh_after_six_hours(rec, clock):
    assert rec.needs_refresh()
    rec.refresh()
    assert not rec.needs_refresh()
    clock.now += 6 * 3600 + 1
    assert rec.needs_refresh()


def test_dismiss_removes_and_pulls_profile(rec):
    rec.refresh()
    first = rec.suggestions()[0]
    rec.dismiss(first)
    assert first.candidate.video_id not in [x.candidate.video_id for x in rec.suggestions()]
    assert any(d.weight < 0 for d in rec.profile().docs)


def test_blocked_channel_filtered_and_undoable(rec):
    rec.refresh()
    ch = rec.suggestions()[0].candidate.channel
    rec.block_channel(ch)
    assert ch not in [x.candidate.channel for x in rec.suggestions()] and rec.blocked_channels() == [ch]
    rec.unblock_channel(ch)
    assert rec.blocked_channels() == []


def test_downloading_a_suggestion_is_logged_once(rec):
    rec.refresh()
    s = rec.suggestions()[0]
    rec.store.add(Download(url=s.candidate.url, kind=Kind.VIDEO, filename="f", folder="d",
                           title=s.candidate.title, video_id=s.candidate.video_id))
    rec.suggestions(); rec.suggestions()
    n = rec.store.query("SELECT COUNT(*) FROM rec_events WHERE event = 'downloaded'")[0][0]
    assert n == 1


def test_refresh_offline_keeps_old_cache(rec, store):
    rec.refresh()
    before = rec.log.candidates()
    offline = Recommender(store, search=lambda q, l: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(DownloadError):
        offline.refresh()
    assert offline.log.candidates() == before


def test_no_suggestions_before_three_downloads(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    assert Recommender(store, search=_fake_search).suggestions() == []


def test_default_search_is_read_when_built(store, monkeypatch):
    # The default is looked up at construction, so tests (and the UI conftest) can stub youtube.search.
    calls = []
    monkeypatch.setattr("downloader.engine.youtube.search", lambda q, l: _fake_search(q, l, calls))
    Recommender(store).refresh()
    assert calls


def test_refresh_does_not_search_before_three_downloads(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    calls = []
    Recommender(store, search=lambda q, l: _fake_search(q, l, calls)).refresh()
    assert calls == []


# ---- whole-branch review fixes -------------------------------------------------------------

def _counting_all(store):
    calls = []
    real = store.all
    store.all = lambda: (calls.append(1), real())[1]
    return calls


def test_view_reads_downloads_once(rec):
    rec.refresh()
    calls = _counting_all(rec.store)
    v = rec.view(words=15)
    assert v.ready and v.suggestions and v.words and len(calls) == 1


def test_a_second_refresh_while_one_runs_does_not_search(store):
    import threading
    entered, release, calls = threading.Event(), threading.Event(), []

    def slow(q, limit):
        calls.append(q)
        entered.set()
        release.wait(5)
        return _fake_search(q, limit)

    rec = Recommender(store, search=slow)
    t = threading.Thread(target=rec.refresh)
    t.start()
    assert entered.wait(5)
    rec.refresh()  # returns at once: a refresh is already running
    n = len(calls)
    release.set()
    t.join(5)
    assert n == 1


def test_shown_is_logged_once_per_video_per_refresh(rec):
    rec.refresh()
    first = rec.suggestions()
    rec.dismiss(first[0])
    rec.suggestions()  # a Dismiss re-ranks: videos that newly enter the list are new impressions
    count = lambda: rec.store.query("SELECT COUNT(*) FROM rec_events WHERE event = 'shown'")[0][0]
    after_dismiss = count()
    rec.suggestions()  # a plain redraw is not
    rows = rec.store.query("SELECT video_id FROM rec_events WHERE event = 'shown'")
    assert count() == after_dismiss and len(rows) == len({r[0] for r in rows})


def test_blocked_channel_is_not_searched(store):
    calls = []
    rec = Recommender(store, search=lambda q, l: _fake_search(q, l, calls))
    rec.block_channel("TS")
    rec.refresh()
    assert calls and "TS" not in calls


def test_discovery_pick_stays_after_a_dismiss(rec):
    rec.refresh()
    s = rec.suggestions()
    picked = next(x.candidate.video_id for x in s if x.discovery)
    rec.dismiss(s[0])
    assert next(x.candidate.video_id for x in rec.suggestions() if x.discovery) == picked


def test_a_refresh_that_finds_nothing_keeps_the_list(rec, store, clock):
    rec.refresh()
    before = rec.log.candidates()
    clock.now += 7 * 3600
    empty = Recommender(store, search=lambda q, l: [], clock=lambda: clock.now)
    empty.refresh()
    assert empty.log.candidates() == before and not empty.needs_refresh()


def test_events_are_indexed(rec):
    names = [r[0] for r in rec.store.query("SELECT name FROM sqlite_master WHERE type = 'index'")]
    assert "rec_events_event" in names
