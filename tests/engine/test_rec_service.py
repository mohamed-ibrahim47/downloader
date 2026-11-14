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
    assert set(f) == {"similarity", "boost", "affinity", "duration", "rank", "discovery", "shared", "kind",
                      "ranker"}
    assert f["ranker"] == "v1"
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


def test_shown_rows_carry_their_refresh_and_search_kind(rec, clock):
    # v2 groups impressions into the list they were shown in, and learns which kind of search works.
    rec.refresh()
    first = clock.now
    rec.suggestions()
    clock.now += 7 * 3600
    rec.refresh()
    rec.suggestions()
    rows = rec.store.query("SELECT refresh, features FROM rec_events WHERE event = 'shown'")
    assert {r[0] for r in rows} == {first, clock.now}
    assert {json.loads(f)["kind"] for _, f in rows} <= {"channel", "newest", "words"}


def test_an_old_database_gets_the_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "db.sqlite"
    Store(path).close()
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE rec_candidates (video_id TEXT PRIMARY KEY, url TEXT, title TEXT, channel TEXT, "
               "duration INTEGER, query TEXT, found_at REAL)")
    db.execute("CREATE TABLE rec_events (id INTEGER PRIMARY KEY, video_id TEXT, event TEXT, title TEXT, "
               "channel TEXT, score REAL, rank INTEGER, features TEXT, at REAL)")
    db.execute("INSERT INTO rec_candidates VALUES ('v', 'u', 't', 'c', 200, 'q', 1.0)")
    db.commit()
    db.close()
    s = Store(path)
    rec = Recommender(s, search=_fake_search)
    assert rec.log.candidates()[0].kind is None
    assert "refresh" in {r[1] for r in s.query("PRAGMA table_info(rec_events)")}
    s.close()


def test_the_log_feeds_training_rows(rec, clock):
    from downloader.engine.recommend.dataset import rows
    rec.refresh()
    first = rec.suggestions()
    rec.dismiss(first[0])
    clock.now += 7 * 3600
    rec.refresh()
    rec.suggestions()
    events = rec.log.events()
    assert [e[2] for e in events][:len(first)] == ["shown"] * len(first)
    got = rows(events)
    assert len(got) == len(first)  # the first list is settled; the second is still on the page
    assert [r.label for r in got].count(0) == len(first) and got[0].weight == 1.0


def _model(w_similarity, active=True):
    from downloader.engine.recommend.features import NAMES
    from downloader.engine.recommend.learn import Weights
    w = Weights(NAMES, [w_similarity] + [0.0] * 7, 0.0, [0.0] * 8, [1.0] * 8, "hand")
    return {"active": active, "reason": "test", "weights": w.to_json(), "compare": {}, "report": {},
            "trained_at": 0.0}


def test_a_learned_model_ranks_the_page(rec):
    rec.refresh()
    v1 = [x.candidate.video_id for x in rec.suggestions()]
    rec.store.set_setting("rec_model", _model(-5.0))  # a model that likes what v1 dislikes
    learned = rec.suggestions()
    assert [x.candidate.video_id for x in learned] != v1
    assert all(0.0 <= x.score <= 1.0 for x in learned)  # v2's score is a probability
    assert rec.ranker() == "v2"


def test_an_inactive_model_or_the_setting_off_keeps_v1(rec, store):
    from downloader.engine.settings import Settings
    rec.refresh()
    v1 = [x.candidate.video_id for x in rec.suggestions()]
    store.set_setting("rec_model", _model(-5.0, active=False))
    assert [x.candidate.video_id for x in rec.suggestions()] == v1 and rec.ranker() == "v1"
    store.set_setting("rec_model", _model(-5.0))
    settings = Settings.load(store)
    settings.rec_learned = False
    settings.save(store)
    assert [x.candidate.video_id for x in rec.suggestions()] == v1 and rec.ranker() == "v1"


def test_the_ranker_is_logged_with_each_impression(rec, clock):
    rec.refresh()
    rec.store.set_setting("rec_model", _model(-5.0))
    rec.suggestions()
    rows = rec.store.query("SELECT features FROM rec_events WHERE event = 'shown'")
    assert {json.loads(f)["ranker"] for (f,) in rows} == {"v2"}


def test_a_refresh_trains_and_saves_why(rec, clock):
    rec.refresh()
    rec.suggestions()
    clock.now += 7 * 3600
    rec.refresh()
    saved = rec.store.get_setting("rec_model")
    assert saved["active"] is False and saved["reason"].startswith("Learning: 10 of 100 rows")
    assert rec.ranker() == "v1"


def test_train_uses_the_chosen_trainer_and_compares_both(rec, clock, store, monkeypatch):
    pytest.importorskip("sklearn")
    from downloader.engine.recommend import evaluate
    from downloader.engine.settings import Settings
    monkeypatch.setattr(evaluate, "MIN_ROWS", 1)
    monkeypatch.setattr(evaluate, "MIN_DOWNLOADS", 0)
    monkeypatch.setattr(evaluate, "MIN_TEST_LISTS", 0)
    for _ in range(3):
        rec.refresh()
        s = rec.suggestions()
        rec.dismiss(s[1])
        rec.dismiss(s[2])
        store.add(Download(url="u", kind=Kind.VIDEO, filename="f", folder="d", title=s[0].candidate.title,
                           channel=s[0].candidate.channel, video_id=s[0].candidate.video_id))
        rec.suggestions()  # notices the download
        clock.now += 7 * 3600
    settings = Settings.load(store)
    settings.rec_trainer = "sklearn"
    settings.save(store)
    rec.train()
    saved = store.get_setting("rec_model")
    assert json.loads(saved["weights"])["trainer"] == "sklearn"
    assert set(saved["compare"]) == {"hand", "sklearn"}


def _net_model(w_similarity):
    from downloader.engine.recommend.features import NAMES
    from downloader.engine.recommend.neural import Net
    net = Net(NAMES, [[1.0] + [0.0] * 7], [0.0], [w_similarity], 0.0, [0.0] * 8, [1.0] * 8, "tanh", 5)
    return {**_model(0.0), "ranker": "v3", "net": net.to_json()}


def test_a_v3_model_ranks_the_page_and_is_logged(rec):
    rec.refresh()
    v1 = [x.candidate.video_id for x in rec.suggestions()]
    rec.store.set_setting("rec_model", _net_model(-5.0))
    before = rec.store.query("SELECT COUNT(*) FROM rec_events WHERE event = 'shown'")[0][0]
    rec._generation = None  # a new list, so the impressions are logged again
    got = rec.suggestions()
    assert [x.candidate.video_id for x in got] != v1 and all(0.0 <= x.score <= 1.0 for x in got)
    assert rec.ranker() == "v3"
    rows = rec.store.query("SELECT features FROM rec_events WHERE event = 'shown' ORDER BY id")
    assert {json.loads(f)["ranker"] for (f,) in rows[before:]} == {"v3"}


def test_the_starter_ranks_until_the_log_is_big_enough(rec, store, monkeypatch):
    from downloader.engine.recommend import starter
    from downloader.engine.settings import Settings
    rec.refresh()
    v1 = [x.candidate.video_id for x in rec.suggestions()]
    monkeypatch.setattr(starter, "RANKER", "v2")
    weights = _model(-5.0)["weights"]
    monkeypatch.setattr(starter, "model", lambda: __import__(
        "downloader.engine.recommend.learn", fromlist=["Weights"]).Weights.from_json(weights))
    assert rec.ranker() == "starter-v2"  # a fresh log: nothing saved yet
    assert [x.candidate.video_id for x in rec.suggestions()] != v1

    store.set_setting("rec_model", {**_model(0.0, active=False), "report": {"rows": 300, "downloads": 20}})
    assert rec.ranker() == "v1"  # the owner's own log is big enough, and v1 won on it

    store.set_setting("rec_model", {**_model(0.0, active=False), "report": {"rows": 30, "downloads": 2}})
    assert rec.ranker() == "starter-v2"
    settings = Settings.load(store)
    settings.rec_learned = False
    settings.save(store)
    assert rec.ranker() == "v1"


def test_no_starter_means_v1(rec, monkeypatch):
    from downloader.engine.recommend import starter
    monkeypatch.setattr(starter, "RANKER", None)
    assert rec.ranker() == "v1"


def test_train_saves_the_net_when_v3_wins(rec, clock, store, monkeypatch):
    from downloader.engine.recommend import evaluate
    monkeypatch.setattr(evaluate, "choose", lambda rep: ("v3", "v3 beats v2 and v1"))
    for _ in range(3):
        rec.refresh()
        s = rec.suggestions()
        rec.dismiss(s[1])
        store.add(Download(url="u", kind=Kind.VIDEO, filename="f", folder="d", title=s[0].candidate.title,
                           channel=s[0].candidate.channel, video_id=s[0].candidate.video_id))
        rec.suggestions()
        clock.now += 7 * 3600
    rec.train()
    saved = store.get_setting("rec_model")
    assert saved["active"] and saved["ranker"] == "v3" and saved["net"]
    assert "v3" in saved["report"]["ndcg"] and saved["report"]["v3_epochs"] >= 1
    assert rec.ranker() == "v3"
