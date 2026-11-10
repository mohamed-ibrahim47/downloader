import random
import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QToolButton

from downloader.engine.models import Download, Kind
from downloader.engine.recommend.service import REFRESH_AFTER, Recommender
from downloader.engine.store import Store
from downloader.engine.youtube import SearchHit
from downloader.ui.for_you_page import ForYouPage


def _hits(q, title=None):
    key = q.replace(" ", "_")
    return [SearchHit(f"{key}{i:02}", f"https://www.youtube.com/watch?v={key}{i:02}",
                      title or f"{q} song {i}", f"{q} ch{i % 5}", 200) for i in range(20)]


def _failing(q, limit):
    raise RuntimeError("offline")


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
def calls():
    return []


@pytest.fixture
def rec(store, clock, calls):
    def search(q, limit):
        calls.append(q)
        return _hits(q)
    return Recommender(store, search=search, clock=lambda: clock.now, rng=random.Random(0))


@pytest.fixture
def empty_rec(tmp_path):
    s = Store(tmp_path / "empty.sqlite")
    yield Recommender(s, search=lambda q, l: _hits(q))
    s.close()


@pytest.fixture
def rec_with_html_title(store):
    return Recommender(store, search=lambda q, l: _hits(q, "<b>bold</b> song"), rng=random.Random(0))


def _page(qtbot, recommender):
    page = ForYouPage(recommender)
    qtbot.addWidget(page)
    return page


def _texts(widget):
    return " ".join(label.text() for label in widget.findChildren(QLabel))


def _title(card):
    return card.findChild(QLabel, "title").text()


def test_cold_start_message(qtbot, empty_rec):
    page = _page(qtbot, empty_rec); page.open()
    assert page.state == "cold" and "Download a few videos and suggestions will show up here." in _texts(page)


def test_open_refreshes_then_lists(qtbot, rec):
    page = _page(qtbot, rec); page.open()
    qtbot.waitUntil(lambda: page.state == "list")
    assert len(page.cards()) == 10 and "Because you downloaded" in _texts(page.cards()[0])


def test_open_with_fresh_cache_does_not_search(qtbot, rec, calls):
    rec.refresh(); calls.clear()
    page = _page(qtbot, rec); page.open()
    assert page.state == "list" and calls == []


def test_dismiss_removes_card_without_searching(qtbot, rec, calls):
    rec.refresh(); calls.clear()
    page = _page(qtbot, rec); page.open()
    title = _title(page.cards()[0])
    qtbot.mouseClick(page.cards()[0].findChild(QPushButton, "dismiss"), Qt.LeftButton)
    assert title not in [_title(c) for c in page.cards()] and calls == []


def test_not_this_channel_hides_it_and_can_be_undone(qtbot, rec):
    rec.refresh()
    page = _page(qtbot, rec); page.open()
    channel = page.suggestion(0).candidate.channel
    page.cards()[0].findChild(QToolButton, "more").menu().actions()[0].trigger()
    assert channel not in [page.suggestion(i).candidate.channel for i in range(len(page.cards()))]
    unblock = page.findChild(QToolButton, "blocked").menu().actions()
    assert [a.text() for a in unblock] == [f"Unblock {channel}"]
    unblock[0].trigger()
    assert rec.blocked_channels() == []


def test_download_emits_url(qtbot, rec):
    rec.refresh(); page = _page(qtbot, rec); page.open()
    with qtbot.waitSignal(page.download_requested) as sig:
        qtbot.mouseClick(page.cards()[0].findChild(QPushButton, "download"), Qt.LeftButton)
    assert sig.args[0].startswith("https://www.youtube.com/watch?v=")


def test_offline_keeps_cached_list(qtbot, rec, store, clock):
    rec.refresh()
    # The cache is just over REFRESH_AFTER old, so open() refreshes, and every search fails.
    flaky = Recommender(store, search=_failing, clock=lambda: clock.now + REFRESH_AFTER + 1)
    page = _page(qtbot, flaky); page.open()
    qtbot.waitUntil(lambda: page.offline)
    assert page.state == "list" and "Couldn't reach YouTube." in _texts(page)


def test_late_result_is_ignored(qtbot, rec):
    page = _page(qtbot, rec)
    page.refresh(); first = page._token; page.refresh()
    page._on_refreshed(first, None)  # the older result arrives last
    assert page.state == "loading"
    qtbot.waitUntil(lambda: not page._busy)  # let the real refreshes finish before teardown


def test_titles_are_plain_text(qtbot, rec_with_html_title):
    page = _page(qtbot, rec_with_html_title); page.open()
    qtbot.waitUntil(lambda: page.state == "list")
    label = page.cards()[0].findChild(QLabel, "title")
    assert label.textFormat() == Qt.PlainText and label.text() == "<b>bold</b> song"


def test_reason_says_same_channel_when_no_words_are_shared(qtbot):
    from downloader.engine.recommend.candidates import Candidate
    from downloader.engine.recommend.rank import Suggestion
    from downloader.ui.for_you_page import SuggestionCard
    c = Candidate("v", "u", "Clean", "Taylor Swift", 200, "q")
    card = SuggestionCard(Suggestion(c, 0.33, 0.18, 0.15, "Blank Space", []), False)
    qtbot.addWidget(card)
    assert card.reason.text() == "Because you downloaded Blank Space · same channel"
    card = SuggestionCard(Suggestion(c, 0.3, 0.3, 0.0, "Blank Space", ["blank"]), False)
    qtbot.addWidget(card)
    assert card.reason.text() == "Because you downloaded Blank Space · blank"
    assert card.more.text() == "…"


def test_why_toggle_shows_scores_and_words(qtbot, rec):
    rec.refresh(); page = _page(qtbot, rec); page.open()
    page.show()  # isVisible() is only true for a widget in a shown window
    assert not page.cards()[0].findChild(QLabel, "score").isVisible()
    page.findChild(QCheckBox, "why").setChecked(True)
    assert "score" in _texts(page.cards()[0]).lower() and page.findChild(QLabel, "profileWords").isVisible()
    assert page.cards()[0].findChild(QLabel, "score").isVisible()


# ---- whole-branch review fixes -------------------------------------------------------------

def test_open_while_loading_does_not_start_another_refresh(qtbot, rec, calls):
    page = _page(qtbot, rec); page.open()
    token = page._token
    page.open()  # e.g. Downloads → For you again while the first search runs
    assert page._token == token
    qtbot.waitUntil(lambda: page.state == "list")


def test_cold_after_a_list_shows_the_message(qtbot, rec, store):
    rec.refresh()
    page = _page(qtbot, rec); page.open()
    assert page.state == "list"
    for d in store.all():
        store.delete(d.id)
    page.open()
    message = [l for l in page.findChildren(QLabel) if l.text() == "Download a few videos and suggestions will show up here."]
    assert page.state == "cold" and page.cards() == [] and message and not message[0].isHidden()


def test_a_dismiss_reads_downloads_once(qtbot, rec, store):
    rec.refresh()
    page = _page(qtbot, rec); page.open()
    calls = []
    real = store.all
    store.all = lambda: (calls.append(1), real())[1]
    qtbot.mouseClick(page.cards()[0].findChild(QPushButton, "dismiss"), Qt.LeftButton)
    assert len(calls) == 1


def test_a_result_after_leaving_is_not_drawn_or_logged(qtbot, rec, store):
    page = _page(qtbot, rec); page.open()
    page.leave()
    qtbot.waitUntil(lambda: not page._busy)
    assert page.cards() == [] and store.query("SELECT COUNT(*) FROM rec_events WHERE event = 'shown'")[0][0] == 0
    page.open()  # coming back draws the fresh cache without searching again
    assert page.state == "list" and len(page.cards()) == 10


def test_offline_with_nothing_cached_shows_one_message(qtbot, store, clock):
    flaky = Recommender(store, search=_failing, clock=lambda: clock.now)
    page = _page(qtbot, flaky); page.open()
    qtbot.waitUntil(lambda: page.offline)
    shown = [l.text() for l in page.findChildren(QLabel) if l.text() and not l.isHidden()]
    assert page.state == "empty" and "Couldn't reach YouTube." in shown
    assert "No new suggestions. Try Refresh later." not in shown


def test_cards_drawn_while_the_page_is_shown_are_not_squashed(qtbot, rec):
    # A Dismiss redraws the list while the page is on screen; the cards must keep their full height.
    rec.refresh()
    page = _page(qtbot, rec)
    page.resize(940, 700)
    page.show()
    page.open()
    qtbot.wait(50)
    for card in page.cards():
        assert card.height() >= card.minimumSizeHint().height()
        assert card.height() >= card.sizeHint().height()


def test_updated_uses_the_recommenders_clock(qtbot, rec, clock):
    rec.refresh()
    clock.now += 2 * 3600 + 5
    page = _page(qtbot, rec); page.open()
    assert "Updated 2 h ago" in _texts(page)


def test_why_says_which_ranker_and_how_learning_goes(qtbot, rec, clock):
    rec.refresh(); page = _page(qtbot, rec); page.open()
    clock.now += REFRESH_AFTER + 1
    rec.refresh()  # trains on the first list
    page.show()
    page.findChild(QCheckBox, "why").setChecked(True)
    info = page.findChild(QLabel, "rankerInfo")
    assert info.isVisible() and "Hand-tuned (v1)" in info.text() and "Learning:" in info.text()


def test_a_learned_ranker_shows_its_weights_and_honest_scores(qtbot, rec):
    from downloader.engine.recommend.features import NAMES
    from downloader.engine.recommend.learn import Weights
    rec.refresh()
    hand = Weights(NAMES, [1.5] + [0.0] * 7, -2.0, [0.0] * 8, [1.0] * 8, "hand")
    sk = Weights(NAMES, [1.49] + [0.0] * 7, -2.0, [0.0] * 8, [1.0] * 8, "sklearn")
    rec.store.set_setting("rec_model", {"active": True, "reason": "v2 beats v1", "weights": hand.to_json(),
                                        "compare": {"hand": hand.to_json(), "sklearn": sk.to_json()},
                                        "report": {}, "trained_at": 0.0})
    page = _page(qtbot, rec); page.open(); page.show()
    page.findChild(QCheckBox, "why").setChecked(True)
    text = page.findChild(QLabel, "rankerInfo").text()
    assert "Learned (v2)" in text and "similarity" in text and "1.50" in text and "1.49" in text
    score = page.cards()[0].findChild(QLabel, "score").text()
    assert "chance" in score and "=" not in score  # not v1's sum any more
