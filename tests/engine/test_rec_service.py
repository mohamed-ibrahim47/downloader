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
                      "ranker", "meaning"}
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
    w = Weights(NAMES, [w_similarity] + [0.0] * (len(NAMES) - 1), 0.0, [0.0] * len(NAMES), [1.0] * len(NAMES), "hand")
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
    net = Net(NAMES, [[1.0] + [0.0] * (len(NAMES) - 1)], [0.0], [w_similarity], 0.0, [0.0] * len(NAMES), [1.0] * len(NAMES), "tanh", 5)
    return {**_model(0.0), "ranker": "v3", "net": net.to_json()}


def test_a_v3_model_ranks_the_page_and_is_logged(rec):
    rec.refresh()
    v1 = [x.candidate.video_id for x in rec.suggestions()]
    rec.store.set_setting("rec_model", _net_model(-5.0))
    before = rec.store.query("SELECT COUNT(*) FROM rec_events WHERE event = 'shown'")[0][0]
    rec.log.mark_refreshed(rec.log.last_refresh() + 1)  # a new list, so the impressions are logged again
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
    monkeypatch.setattr(evaluate, "choose", lambda rep, *a: ("v3", "v3 beats v2 and v1"))
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


def test_files_setting_controls_file_downloads_in_profile(rec, store):
    from downloader.engine.settings import Settings
    store.add(Download(url="u", kind=Kind.FILE, filename="Complete_Python_Bootcamp.zip", folder="d"))
    assert "complete python bootcamp" in [d.title for d in rec.profile().docs]
    s = Settings.load(store)
    s.rec_files = False
    s.save(store)
    assert "complete python bootcamp" not in [d.title for d in rec.profile().docs]


def test_new_settings_default_on():
    from downloader.engine.settings import Settings
    assert Settings().rec_meaning is True and Settings().rec_files is True


def test_saved_model_with_old_inputs_is_ignored(rec, store):
    from downloader.engine.recommend.features import NAMES
    from downloader.engine.recommend.learn import Weights
    old = NAMES[:8]
    w = Weights(old, [-5.0] + [0.0] * 7, 0.0, [0.0] * 8, [1.0] * 8, "hand")  # saved by 1.4.0: 8 inputs
    rec.refresh()  # trains, so the old model is saved after it
    store.set_setting("rec_model", {"active": True, "ranker": "v2", "reason": "old", "weights": w.to_json(),
                                    "compare": {}, "report": {}})
    assert rec.ranker() == "v1"
    assert rec.suggestions()


def test_shown_row_logs_meaning(rec):
    rec.refresh()
    rec.suggestions()
    f = json.loads(rec.store.query("SELECT features FROM rec_events WHERE event = 'shown' LIMIT 1")[0][0])
    assert "meaning" in f


def test_events_carry_the_title(rec):
    rec.refresh()
    s = rec.suggestions()
    e = rec.log.events()[0]
    assert len(e) == 7 and e[6] == s[0].candidate.title


# ---- v4: vectors, retraining, the v4 ranker, the banner (design v4, E4/E7/E8) ----------------------

from downloader.engine.recommend import embed, starter_vectors  # noqa: E402
from downloader.engine.recommend.embed import Vectors  # noqa: E402

STARTER = Vectors(["blank", "space", "style", "shake", "song", "piano"],
                  [[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0], [0.7, 0.3, 0], [0.5, 0.5, 0], [0, 0, 1]], [9] * 6)


@pytest.fixture
def with_starter(monkeypatch):
    monkeypatch.setattr(starter_vectors, "vectors", lambda: STARTER)
    monkeypatch.setattr(starter_vectors, "DEFAULT_ON", True)
    monkeypatch.setattr(starter_vectors, "EXAMPLE", ("piano", "acoustic"))


def _add_titles(store, n, start=0):
    for i in range(start, start + n):
        store.add(Download(url="u", kind=Kind.VIDEO, filename="f", folder="d", title=f"song number{i} piano"))


def _set(store, **kw):
    from downloader.engine.settings import Settings
    s = Settings.load(store)
    for k, v in kw.items():
        setattr(s, k, v)
    s.save(store)


def test_fresh_db_uses_starter_vectors(rec, with_starter):
    assert rec.vectors() is STARTER
    assert rec.vectors_meta()["source"] == "starter"


def test_fresh_db_without_starter_has_no_vectors(rec):
    assert rec.vectors() is None and rec.ranker() == "v1"


def test_retrain_waits_for_300_new_titles(rec, store, with_starter):
    _add_titles(store, 296)  # + the fixture's 3 = 299 titles
    assert rec.retrain_vectors() is False
    _add_titles(store, 1, start=296)
    assert rec.retrain_vectors() is True
    meta = rec.vectors_meta()
    assert meta["source"] == "own" and meta["titles"] == 300
    assert rec.retrain_vectors() is False  # nothing new since


def test_retrain_uses_only_user_titles_and_keeps_starter_words(rec, store, with_starter):
    _add_titles(store, 300)
    assert rec.retrain_vectors()
    v = rec.log.load_vectors()
    assert set(STARTER.words) <= set(v.words) and "number5" not in v.words  # seen once: below USER_MIN_COUNT
    assert rec.vectors().words == v.words


def test_retrain_failure_keeps_old_vectors(rec, store, with_starter, monkeypatch):
    _add_titles(store, 300)

    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(embed, "train", boom)
    assert rec.retrain_vectors() is False
    assert rec.log.load_vectors() is None and rec.vectors() is STARTER
    rec.refresh()
    assert rec.suggestions()


def test_retrains_never_overlap(rec, store, with_starter):
    _add_titles(store, 300)
    with rec._retraining:
        assert rec.retrain_vectors() is False


def test_v4_ranks_when_default_on(rec, with_starter):
    from downloader.engine.recommend.meaning import v4_score
    rec.refresh()
    assert rec.ranker() == "v4"
    s = rec.suggestions(20)
    assert any(x.meaning > 0 for x in s)
    plain = [x for x in s if not x.discovery]
    scores = [v4_score(x.similarity, x.meaning, x.boost / 0.15, starter_vectors.MEANING_MIX) for x in plain]
    assert scores == sorted(scores, reverse=True) or len({x.candidate.channel for x in plain}) > 1
    assert all(x.score == pytest.approx(sc) for x, sc in zip(plain, scores))


def test_meaning_off_means_v1_base_and_zero_meaning(rec, store, with_starter):
    _set(store, rec_meaning=False)
    rec.refresh()
    assert rec.ranker() == "v1"
    rec.suggestions()
    logged = [json.loads(r[0])["meaning"] for r in store.query("SELECT features FROM rec_events WHERE event='shown'")]
    assert logged and all(m == 0 for m in logged)
    assert not rec.uses_meaning()


def test_learned_off_still_matches_by_meaning(rec, store, with_starter):
    _set(store, rec_learned=False)
    rec.refresh()
    assert rec.ranker() == "v4"


def test_training_recomputes_meaning_for_old_rows(rec, with_starter, monkeypatch):
    from downloader.engine.recommend import service
    seen = {}
    real = service.training_rows

    def spy(events, **kw):
        seen.update(kw)
        return real(events, **kw)
    monkeypatch.setattr(service, "training_rows", spy)
    rec.refresh()
    rec.suggestions()
    rec.train()
    assert callable(seen.get("meaning")) and seen["meaning"]("blank space piano", rec._clock() + 1) > 0  # downloads before it count


def test_banner_due_once(rec, with_starter):
    rec.refresh()
    assert rec.banner_due() and rec.example() == ("piano", "acoustic")
    rec.banner_done()
    assert not rec.banner_due()


def test_no_banner_without_meaning(rec):
    rec.refresh()
    assert not rec.banner_due()


def test_nearest_words_for_profile(rec, with_starter):
    rec.refresh()
    got = dict(rec.nearest_words(5, 2))
    assert got["blank"] == ["space", "style"]


def test_refresh_does_not_retrain_vectors(rec, store, with_starter):
    """The word2vec retrain takes minutes in pure Python: it runs as its own task after the list is up, not
    inside refresh/train, which would keep the page on "Looking for new videos..." (final review, finding 2)."""
    _add_titles(store, 300)
    rec.refresh()
    rec.train()
    assert rec.vectors_meta()["source"] == "starter"
    assert rec.retrain_vectors() is True


def test_meaning_index_is_reused_between_draws(rec, with_starter, monkeypatch):
    """view() runs on the GUI thread at every draw, dismissal and Why toggle: the meaning index is built
    again only when what it depends on changes (final review, finding 3)."""
    from downloader.engine.recommend import service
    built = []
    real = service.MeaningIndex
    monkeypatch.setattr(service, "MeaningIndex", lambda *a: built.append(1) or real(*a))
    rec.refresh()
    first = rec.suggestions()
    rec.suggestions()
    rec.view(words=5)
    assert len(built) == 1
    rec.dismiss(first[0])
    after = rec.suggestions()
    assert len(built) == 2 and first[0].candidate.video_id not in {s.candidate.video_id for s in after}


# ---- v5: the discovery card's bandit ----------------------------------------------------

from downloader.engine.recommend import service as service_mod  # noqa: E402
from downloader.engine.recommend.bayes import Posterior  # noqa: E402
from downloader.engine.recommend.service import BANDIT_KEY, PICK_KEY  # noqa: E402
from downloader.engine.settings import Settings  # noqa: E402


def _set_mode(store, mode):
    s = Settings.load(store)
    s.rec_discovery = mode
    s.save(store)


def _shown_features(store):
    return [json.loads(r[0]) for r in store.query("SELECT features FROM rec_events WHERE event = 'shown' ORDER BY id")]


def _disc(picks):
    return next(s for s in picks if s.discovery)


def test_discovery_row_logs_picker_and_propensity(rec):
    rec.refresh()
    rec.suggestions()
    rows = _shown_features(rec.store)
    disc = [f for f in rows if f["discovery"]]
    assert len(disc) == 1 and disc[0]["picker"] == "source" and 0 < disc[0]["propensity"] <= 1
    assert disc[0]["fallback"] is None
    assert all("picker" not in f for f in rows if not f["discovery"])


def test_random_mode_logs_one_over_pool(rec, store):
    _set_mode(store, "random")
    rec.refresh()
    rec.suggestions()
    disc = [f for f in _shown_features(store) if f["discovery"]][0]
    assert disc["picker"] == "random" and disc["fallback"] is None
    n = 1 / disc["propensity"]
    assert n == pytest.approx(round(n)) and 1 <= round(n) <= 20


def test_restart_keeps_pick_and_does_not_relog(rec, store, clock):
    rec.refresh()
    first = _disc(rec.suggestions()).candidate.video_id
    again = Recommender(store, search=_fake_search, clock=lambda: clock.now, rng=random.Random(99))
    assert _disc(again.suggestions()).candidate.video_id == first
    assert len(_shown_features(store)) == 10


def test_a_words_only_view_draws_but_the_logged_row_still_has_its_pick(rec, store):
    rec.refresh()
    rec.words()  # draws the card without logging it
    rec.suggestions()
    disc = [f for f in _shown_features(store) if f["discovery"]]
    assert len(disc) == 1 and disc[0]["picker"] == "source"


def test_mode_change_keeps_card_until_next_draw(rec, store):
    rec.refresh()
    card = _disc(rec.suggestions())
    _set_mode(store, "contextual")
    assert _disc(rec.suggestions()).candidate.video_id == card.candidate.video_id
    rec.dismiss(card)
    new = _disc(rec.suggestions())
    assert new.candidate.video_id != card.candidate.video_id
    disc = [f for f in _shown_features(store) if f["discovery"]]
    assert len(disc) == 2
    assert disc[1]["picker"] == "contextual" or (disc[1]["picker"] == "random" and disc[1]["fallback"] == "contextual")


def test_contextual_without_posterior_falls_back(rec, store):
    _set_mode(store, "contextual")
    rec.refresh()
    store.set_setting(BANDIT_KEY, None)
    rec.suggestions()
    disc = [f for f in _shown_features(store) if f["discovery"]][0]
    assert disc["picker"] == "random" and disc["fallback"] == "contextual"


def test_contextual_with_stale_posterior_falls_back(rec, store):
    _set_mode(store, "contextual")
    rec.refresh()
    stale = Posterior(["x"], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [0.0], [1.0], 0.0)
    store.set_setting(BANDIT_KEY, {"posterior": stale.to_json(), "error": None, "trained_at": 0})
    rec.suggestions()
    disc = [f for f in _shown_features(store) if f["discovery"]][0]
    assert disc["picker"] == "random" and disc["fallback"] == "contextual"


def test_contextual_mode_picks_with_the_trained_posterior(rec, store):
    _set_mode(store, "contextual")
    rec.refresh()
    rec.suggestions()
    disc = [f for f in _shown_features(store) if f["discovery"]][0]
    assert disc["picker"] == "contextual" and 0 < disc["propensity"] <= 1


def test_train_saves_a_posterior_from_discovery_rows(rec, store, clock):
    rec.refresh()
    rec.dismiss(_disc(rec.suggestions()))
    rec.suggestions()
    clock.now += 7 * 3600
    rec.refresh()  # the first list is settled now: its two discovery cards are rows
    saved = store.get_setting(BANDIT_KEY)
    assert saved["error"] is None and Posterior.from_json(saved["posterior"]).rows > 1


def test_a_bandit_training_error_is_saved_not_raised(rec, store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad maths")
    monkeypatch.setattr(service_mod, "posterior_from_rows", boom)
    rec.refresh()
    saved = store.get_setting(BANDIT_KEY)
    assert saved["posterior"] is None and "bad maths" in saved["error"]
    assert store.get_setting(service_mod.MODEL_KEY) is not None  # the ranker's training still saved


def test_discovery_info_reports_mode_arms_and_pick(rec, store):
    rec.refresh()
    card = _disc(rec.suggestions())
    info = rec.discovery_info()
    assert info["mode"] == "source" and set(info["arms"]) == {"channel", "newest", "words"}
    assert info["pick"]["video_id"] == card.candidate.video_id and info["pick"]["picker"] == "source"
    assert info["pick"]["draws"] and info["posterior"] is not None and info["error"] is None
    assert store.get_setting(PICK_KEY)["refresh"] == rec.log.last_refresh()


def test_arms_are_not_recomputed_while_the_log_is_unchanged(rec, monkeypatch):
    # discovery_info() runs on every render with "Why this order?" ticked: reading the whole log each time
    # froze the GUI on a big log.
    rec.refresh()
    rec.suggestions()
    reads = []
    real = rec.log.events
    monkeypatch.setattr(rec.log, "events", lambda: reads.append(1) or real())
    first = rec.discovery_info()["arms"]
    rec.discovery_info()
    assert len(reads) == 1
    rec.dismiss(_disc(rec.suggestions()))  # a new event: the arms must see it
    rec.discovery_info()
    assert len(reads) >= 2 and set(first) == {"channel", "newest", "words"}


# ---- v7: the two-tower ranker behind its setting (design v7, A1, A4, A7) -------------------------

from downloader.engine.recommend import twotower  # noqa: E402
from downloader.engine.recommend.twotower import TwoTower  # noqa: E402


def _tower_json(vectors_at=None):
    # d = 3 (STARTER's size), k = 2: likes the piano direction, dislikes blank/space
    m = TwoTower([[-1.0, 0.0, 2.0], [0.0, -1.0, 1.0]], [0.0, 0.0], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], [0.5, 0.5],
                 vectors_at=vectors_at)
    return m.to_json()


def _save_v7(rec, store, **kw):
    saved = rec.model()
    store.set_setting("rec_two_tower_model", {"trained_at": saved["trained_at"],
                                              "vectors_at": rec.vectors_meta().get("at"), "active": True,
                                              "reason": "v7 beats v4", "ndcg": {"v7": 0.9, "v4": 0.5}, "lists": 3,
                                              "model": _tower_json(), **kw})


def test_two_tower_setting_default_off():
    from downloader.engine.settings import Settings
    assert Settings().rec_two_tower is False


def test_setting_off_never_calls_twotower(rec, store, clock, with_starter, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("twotower ran with the setting off")
    for name in ("contest", "fit", "examples", "history_of"):
        monkeypatch.setattr(twotower, name, boom)
    monkeypatch.setattr(TwoTower, "from_json", boom)
    rec.refresh()
    rec.suggestions()
    clock.now += 7 * 3600
    rec.refresh()
    rec.suggestions()
    assert store.get_setting("rec_two_tower_model") is None
    assert rec.two_tower_info() is None


def test_v7_ranks_when_saved_and_current(rec, store, with_starter):
    _set(store, rec_two_tower=True)
    rec.refresh()  # trains: rec_model gets its trained_at
    _save_v7(rec, store)
    assert rec.ranker() == "v7"
    picks = rec.suggestions()
    assert {f["ranker"] for f in _shown_features(store)} == {"v7"}
    assert rec.two_tower_info()["reason"] == "v7 beats v4"
    assert all(-1.0 <= x.score <= 1.0 for x in picks)  # u·v with k = 2 tanh outputs is within ±2; here ±1


@pytest.mark.parametrize("off", ["rec_two_tower", "rec_learned", "rec_meaning"])
def test_v7_off_at_once(rec, store, with_starter, off):
    _set(store, rec_two_tower=True)
    rec.refresh()
    _save_v7(rec, store)
    assert rec.ranker() == "v7"
    _set(store, **{off: False})
    assert rec.ranker() != "v7"


def test_v7_steps_aside_after_vectors_retrain(rec, store, with_starter):
    _set(store, rec_two_tower=True)
    rec.refresh()
    _save_v7(rec, store)
    assert rec.ranker() == "v7"
    store.set_setting("rec_vectors_meta", {"source": "own", "words": 6, "titles": 300, "at": 12345.0})
    assert rec.ranker() != "v7"


def test_v7_stale_after_newer_training(rec, store, with_starter):
    _set(store, rec_two_tower=True)
    rec.refresh()
    _save_v7(rec, store)
    assert rec.ranker() == "v7"
    saved = rec.model()
    store.set_setting("rec_model", {**saved, "trained_at": saved["trained_at"] + 1})
    assert rec.ranker() != "v7"


def test_v7_training_error_keeps_main_decision(rec, store, with_starter, monkeypatch):
    _set(store, rec_two_tower=True)

    def boom(*a, **k):
        raise RuntimeError("bad batch")
    monkeypatch.setattr(twotower, "contest", boom)
    rec.refresh()
    assert rec.model()["reason"].startswith("Learning")
    info = rec.two_tower_info()
    assert info["active"] is False and info["reason"] == "Training failed: bad batch"
    assert info["trained_at"] == rec.model()["trained_at"]
    assert rec.suggestions()


def test_v7_without_meaning_says_why(rec, store, with_starter):
    _set(store, rec_two_tower=True, rec_meaning=False)
    rec.refresh()
    info = rec.two_tower_info()
    assert info["active"] is False and info["reason"] == "v7 needs Match by meaning"


def test_v7_small_log_reason(rec, store, with_starter):
    _set(store, rec_two_tower=True)
    rec.refresh()
    assert rec.two_tower_info()["reason"].startswith("Learning: 0 of 100 rows")


def test_old_database_setting_on(rec, store, with_starter):
    store.set_setting("rec_model", _model(-5.0))  # saved by 1.6.0: no v7 key
    _set(store, rec_two_tower=True)
    assert rec.ranker() == "v2"
    assert rec.two_tower_info() is None
    rec.suggestions()  # no candidates yet (no refresh): the point is that nothing raises


def test_v7_cards_are_not_shown_as_a_chance(rec, store, with_starter):
    _set(store, rec_two_tower=True)
    rec.refresh()
    _save_v7(rec, store)
    picks = rec.suggestions()
    assert picks and all(x.two_tower and not x.learned for x in picks)


def test_v7_stamp_is_the_vectors_it_trained_on(rec, store, with_starter, monkeypatch):
    """The word vectors retrain (their own task) while train() runs: v7 must carry the stamp of the vectors it
    was fit on, so it steps aside instead of ranking with vectors it was never judged on."""
    from downloader.engine.recommend import evaluate
    _set(store, rec_two_tower=True)
    real = evaluate.evaluate

    def retrain_meanwhile(*a, **k):
        store.set_setting("rec_vectors_meta", {"source": "own", "words": 6, "titles": 300, "at": 999.0})
        return real(*a, **k)
    monkeypatch.setattr(evaluate, "evaluate", retrain_meanwhile)
    monkeypatch.setattr(twotower, "contest", lambda *a, **k: twotower.Contest(3, 0.9, 0.5, True, "v7 beats v4"))
    rec.refresh()
    info = rec.two_tower_info()
    assert info["active"] and info["vectors_at"] is None
    assert rec.ranker() != "v7"


# ---- v8: the sequence ranker behind its setting (design v8, A1, A4) -------------------------------

import random as _random  # noqa: E402

from downloader.engine.recommend import sasrec  # noqa: E402
from downloader.engine.recommend.sasrec import SasRec  # noqa: E402


def _sasrec_json():
    return sasrec.init(d=3, k=2, L=4, rng=_random.Random(0)).to_json()  # d = 3: STARTER's size


def _save_v8(rec, store, **kw):
    saved = rec.model()
    store.set_setting("rec_sasrec_model", {"trained_at": saved["trained_at"],
                                           "vectors_at": rec.vectors_meta().get("at"), "active": True,
                                           "reason": "v8 beats v4", "ndcg": {"v8": 0.9, "v4": 0.5}, "lists": 3,
                                           "rival": "v4", "model": _sasrec_json(), **kw})


def test_sasrec_setting_default_off():
    from downloader.engine.settings import Settings
    assert Settings().rec_sasrec is False


def test_setting_off_never_calls_sasrec(rec, store, clock, with_starter, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("sasrec ran with the setting off")
    for name in ("contest", "fit", "examples"):
        monkeypatch.setattr(sasrec, name, boom)
    monkeypatch.setattr(SasRec, "from_json", boom)
    rec.refresh()
    rec.suggestions()
    clock.now += 7 * 3600
    rec.refresh()
    rec.suggestions()
    assert store.get_setting("rec_sasrec_model") is None
    assert rec.sasrec_info() is None


def test_v8_ranks_when_saved_and_current(rec, store, with_starter):
    _add_titles(store, 3)
    _set(store, rec_sasrec=True)
    rec.refresh()
    _save_v8(rec, store)
    assert rec.ranker() == "v8"
    view = rec.view()
    assert view.suggestions and all(x.two_tower and not x.learned for x in view.suggestions)
    assert {f["ranker"] for f in _shown_features(store)} == {"v8"}
    titles = {d.title for d in store.all()}
    assert view.leaned_on in titles
    assert rec.sasrec_info()["reason"] == "v8 beats v4"


def test_v8_before_v7(rec, store, with_starter):
    _set(store, rec_sasrec=True, rec_two_tower=True)
    rec.refresh()
    _save_v7(rec, store)
    _save_v8(rec, store, rival="v7")
    assert rec.ranker() == "v8"
    _set(store, rec_two_tower=False)  # v8 beat v7, which beat the pick, on the same lists
    assert rec.ranker() == "v8"


@pytest.mark.parametrize("off", ["rec_sasrec", "rec_learned", "rec_meaning"])
def test_v8_off_at_once(rec, store, with_starter, off):
    _set(store, rec_sasrec=True)
    rec.refresh()
    before = rec.ranker()
    _save_v8(rec, store)
    assert rec.ranker() == "v8"
    _set(store, **{off: False})
    assert rec.ranker() != "v8"
    if off == "rec_sasrec":
        assert rec.ranker() == before  # 1.7.0's answer, no retraining


def test_v8_steps_aside_after_vectors_retrain(rec, store, with_starter):
    _set(store, rec_sasrec=True)
    rec.refresh()
    _save_v8(rec, store)
    assert rec.ranker() == "v8"
    store.set_setting("rec_vectors_meta", {"source": "own", "words": 6, "titles": 300, "at": 12345.0})
    assert rec.ranker() != "v8"


def test_v8_stale_after_newer_training(rec, store, with_starter):
    _set(store, rec_sasrec=True)
    rec.refresh()
    _save_v8(rec, store)
    assert rec.ranker() == "v8"
    saved = rec.model()
    store.set_setting("rec_model", {**saved, "trained_at": saved["trained_at"] + 1})
    assert rec.ranker() != "v8"


def test_v8_training_error_keeps_other_decisions(rec, store, with_starter, monkeypatch):
    _set(store, rec_sasrec=True, rec_two_tower=True)

    def boom(*a, **k):
        raise RuntimeError("bad step")
    monkeypatch.setattr(sasrec, "contest", boom)
    rec.refresh()
    assert rec.model()["reason"].startswith("Learning")
    assert rec.two_tower_info()["reason"].startswith("Learning")
    info = rec.sasrec_info()
    assert info["active"] is False and info["reason"] == "Training failed: bad step"
    assert info["trained_at"] == rec.model()["trained_at"]
    assert rec.suggestions()


def test_v8_without_meaning_says_why(rec, store, with_starter):
    _set(store, rec_sasrec=True, rec_meaning=False)
    rec.refresh()
    info = rec.sasrec_info()
    assert info["active"] is False and info["reason"] == "v8 needs Match by meaning"


def test_v8_rival_is_v7_when_v7_won(rec, store, with_starter, monkeypatch):
    _set(store, rec_sasrec=True, rec_two_tower=True)
    split7 = twotower.init(3, 2, _random.Random(0))
    monkeypatch.setattr(twotower, "contest",
                        lambda *a, **k: twotower.Contest(3, 0.9, 0.5, True, "v7 beats v4", model=split7))
    seen = {}

    def fake(rows, history, E, rival, models, v7=None):
        seen.update(rival=rival, v7=v7)
        return sasrec.Contest(3, 0.4, 0.9, False, "v8 didn't beat v7 yet (NDCG 0.40 vs 0.90)")
    monkeypatch.setattr(sasrec, "contest", fake)
    rec.refresh()
    assert seen == {"rival": "v7", "v7": split7}
    assert rec.sasrec_info()["rival"] == "v7"


def test_old_database_sasrec_on(rec, store, with_starter):
    store.set_setting("rec_model", _model(-5.0))  # saved by 1.7.0: no v8 key
    _set(store, rec_sasrec=True)
    assert rec.ranker() == "v2"
    assert rec.sasrec_info() is None
    rec.suggestions()


def test_refresh_can_leave_v7_v8_for_later(rec, store, with_starter, monkeypatch):
    """Review fix: the page shows the new list before v7/v8 train (they can take minutes on a big log)."""
    _set(store, rec_sasrec=True, rec_two_tower=True)
    calls = []
    real7, real8 = twotower.contest, sasrec.contest
    monkeypatch.setattr(twotower, "contest", lambda *a, **k: calls.append("v7") or real7(*a, **k))
    monkeypatch.setattr(sasrec, "contest", lambda *a, **k: calls.append("v8") or real8(*a, **k))
    rec.refresh(experimental=False)
    assert calls == [] and rec.model() is not None
    assert store.get_setting("rec_sasrec_model") is None
    rec.train_experimental()
    assert calls == ["v7", "v8"]
    assert rec.sasrec_info()["trained_at"] == rec.two_tower_info()["trained_at"] == rec.model()["trained_at"]
    rec.train_experimental()  # nothing pending any more: no second training
    assert calls == ["v7", "v8"]


def test_train_experimental_with_nothing_pending(rec, store, with_starter):
    _set(store, rec_sasrec=True)
    rec.train_experimental()
    assert store.get_setting("rec_sasrec_model") is None


def test_v8_view_parses_once_and_reads_history_once(rec, store, with_starter, monkeypatch):
    """Review minor: the saved model is parsed once while it stays the same, and view() runs one forward pass."""
    _add_titles(store, 3)
    _set(store, rec_sasrec=True)
    rec.refresh()
    _save_v8(rec, store)
    parsed, forwards = [], []
    real_parse, real_fwd = SasRec.from_json.__func__, sasrec._forward
    monkeypatch.setattr(SasRec, "from_json", classmethod(lambda cls, t: parsed.append(1) or real_parse(cls, t)))
    monkeypatch.setattr(sasrec, "_forward", lambda *a: forwards.append(1) or real_fwd(*a))
    assert rec.ranker() == "v8"
    view = rec.view()
    rec.ranker()
    assert view.suggestions and len(parsed) == 1 and len(forwards) == 1


def test_leaned_on_matches_view_and_is_none_without_v8(rec, store, with_starter):
    """Review minor: "Leaned most on" is computed when asked, so it can't be stale after a training."""
    _add_titles(store, 3)
    _set(store, rec_sasrec=True)
    rec.refresh()
    assert rec.leaned_on() is None  # v8 not ranking yet
    _save_v8(rec, store)
    assert rec.leaned_on() == rec.view().leaned_on is not None


def test_v7_and_v8_land_together(rec, store, with_starter, monkeypatch):
    """Review minor: while v8 still trains, a v7 that just won must not rank yet, so the page never steps
    pick -> v7 -> v8 within one training run. Both results are saved together at the end."""
    _set(store, rec_sasrec=True, rec_two_tower=True)
    split7 = twotower.init(3, 2, _random.Random(0))
    monkeypatch.setattr(twotower, "contest",
                        lambda *a, **k: twotower.Contest(3, 0.9, 0.5, True, "v7 beats v4", model=split7))
    seen = []

    def fake(rows, history, E, rival, models, v7=None):
        seen.append((rival, rec.ranker()))
        return sasrec.Contest(3, 0.4, 0.9, False, "v8 didn't beat v7 yet (NDCG 0.40 vs 0.90)")
    monkeypatch.setattr(sasrec, "contest", fake)
    rec.refresh()
    assert seen and seen[0][0] == "v7" and seen[0][1] != "v7"
    assert rec.ranker() == "v7"
