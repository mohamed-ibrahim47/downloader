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


def test_refresh_does_not_search_before_three_downloads(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    calls = []
    Recommender(store, search=lambda q, l: _fake_search(q, l, calls)).refresh()
    assert calls == []
