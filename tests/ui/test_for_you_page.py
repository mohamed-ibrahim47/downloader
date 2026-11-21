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
    hand = Weights(NAMES, [1.5] + [0.0] * (len(NAMES) - 1), -2.0, [0.0] * len(NAMES), [1.0] * len(NAMES), "hand")
    sk = Weights(NAMES, [1.49] + [0.0] * (len(NAMES) - 1), -2.0, [0.0] * len(NAMES), [1.0] * len(NAMES), "sklearn")
    rec.store.set_setting("rec_model", {"active": True, "reason": "v2 beats v1", "weights": hand.to_json(),
                                        "compare": {"hand": hand.to_json(), "sklearn": sk.to_json()},
                                        "report": {}, "trained_at": 0.0})
    page = _page(qtbot, rec); page.open(); page.show()
    page.findChild(QCheckBox, "why").setChecked(True)
    text = page.findChild(QLabel, "rankerInfo").text()
    assert "Learned (v2)" in text and "similarity" in text and "1.50" in text and "1.49" in text
    score = page.cards()[0].findChild(QLabel, "score").text()
    assert "chance" in score and "=" not in score  # not v1's sum any more


def test_ranker_text_for_v3_shows_the_network_and_the_scores():
    from downloader.engine.recommend.features import NAMES
    from downloader.engine.recommend.neural import Net
    from downloader.ui.for_you_page import ranker_text
    net = Net(NAMES, [[0.1] * len(NAMES)] * 8, [0.0] * 8, [0.2] * 8, 0.0, [0.0] * len(NAMES), [1.0] * len(NAMES), "relu", 12)
    model = {"active": True, "ranker": "v3", "reason": "v3 beats v2 and v1", "net": net.to_json(), "weights": None,
             "compare": {}, "report": {"ndcg": {"baseline": 0.4, "v1": 0.58, "v2": 0.6, "v3": 0.63}}}
    text = ranker_text(model, "v3")
    assert "Learned (v3), small neural network (9 → 8 → 1, ReLU)" in text
    assert "v1 0.58 · v2 0.60 · v3 0.63" in text


def test_ranker_text_for_the_starter_says_how_far_the_own_log_is():
    from downloader.ui.for_you_page import ranker_text
    model = {"active": False, "reason": "Learning: 30 of 100 rows, 2 of 10 downloads", "compare": {},
             "report": {"rows": 30, "downloads": 2}}
    text = ranker_text(model, "starter-v2")
    assert "Starter (v2), trained on simulated months" in text and "30 of 100 rows" in text
    assert "Hand-tuned" not in text


# ---- v4: the banner, v4 in "Why this order?", nearest words (design v4, E8) ------------------------

from downloader.engine.recommend import starter_vectors  # noqa: E402
from downloader.engine.recommend.embed import Vectors  # noqa: E402
from downloader.ui.for_you_page import nearest_text, ranker_text  # noqa: E402

STARTER = Vectors(["blank", "space", "style", "shake", "song", "piano"],
                  [[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0], [0.7, 0.3, 0], [0.5, 0.5, 0], [0, 0, 1]], [9] * 6)


@pytest.fixture
def meaning_rec(rec, monkeypatch):
    monkeypatch.setattr(starter_vectors, "vectors", lambda: STARTER)
    monkeypatch.setattr(starter_vectors, "DEFAULT_ON", True)
    monkeypatch.setattr(starter_vectors, "EXAMPLE", ("piano cover", "acoustic version"))
    return rec


def _open(qtbot, rec):
    rec.refresh()
    page = _page(qtbot, rec)
    page.open()
    return page


def test_banner_shows_once_with_example(qtbot, meaning_rec):
    page = _open(qtbot, meaning_rec)
    assert page.banner.isVisibleTo(page)
    assert ('For you now matches by meaning, not just words: "piano cover" can bring up "acoustic version".'
            in _texts(page.banner))
    buttons = [b.text() for b in page.banner.findChildren(QPushButton)]
    assert buttons == ["Got it", "Turn off"]


def test_banner_without_example(qtbot, meaning_rec, monkeypatch):
    monkeypatch.setattr(starter_vectors, "EXAMPLE", None)
    page = _open(qtbot, meaning_rec)
    assert _texts(page.banner).strip() == "For you now matches by meaning, not just words."


def test_no_banner_without_meaning(qtbot, rec):
    page = _open(qtbot, rec)
    assert not page.banner.isVisibleTo(page)


def test_got_it_hides_and_marks_done(qtbot, meaning_rec):
    page = _open(qtbot, meaning_rec)
    [b for b in page.banner.findChildren(QPushButton) if b.text() == "Got it"][0].click()
    assert not page.banner.isVisibleTo(page) and not meaning_rec.banner_due()
    page._render()
    assert not page.banner.isVisibleTo(page)


def test_turn_off_asks_to_clear_the_setting_and_hides(qtbot, meaning_rec):
    page = _open(qtbot, meaning_rec)
    with qtbot.waitSignal(page.meaning_off_requested, timeout=1000):
        [b for b in page.banner.findChildren(QPushButton) if b.text() == "Turn off"][0].click()
    assert not page.banner.isVisibleTo(page) and not meaning_rec.banner_due()


def test_ranker_text_v4():
    model = {"active": False, "ranker": "v4", "base": "v4", "reason": "v4 beats v1", "compare": {},
             "report": {"ndcg": {"v1": 0.58, "v4": 0.61, "v2": 0.6, "v3": 0.55}}}
    own = {"source": "own", "words": 7812, "titles": 412, "at": time.mktime((2026, 11, 20, 12, 0, 0, 0, 0, -1))}
    text = ranker_text(model, "v4", own)
    assert ("Ranked by: Meaning (v4), word embeddings: 7,812 words × 32 · retrained on 412 of your titles, "
            "2026-11-20") in text
    assert "NDCG@10 on your newest lists: v1 0.58 · v4 0.61 · v2 0.60 · v3 0.55" in text
    starter = ranker_text(model, "v4", {"source": "starter", "words": 8000, "titles": 0, "at": None})
    assert "Meaning (v4), word embeddings: 8,000 words × 32 · starter, trained on public titles" in starter


def test_nearest_words_lines():
    text = nearest_text([("piano", ["cover", "acoustic", "guitar", "chords"]), ("lion", ["wildlife"])])
    assert text == "Nearest words:\npiano → cover, acoustic, guitar, chords\nlion → wildlife"
    assert nearest_text([]) == ""


def test_why_shows_nearest_words(qtbot, meaning_rec):
    page = _open(qtbot, meaning_rec)
    page._why.setChecked(True)
    assert "Nearest words:" in _texts(page) and "blank → " in _texts(page)


def test_v4_card_score_names_meaning(qtbot, meaning_rec):
    page = _open(qtbot, meaning_rec)
    page._why.setChecked(True)
    text = page.cards()[0].findChild(QLabel, "score").text()
    assert "meaning" in text and "chance" not in text


def test_no_ndcg_line_before_any_test_list():
    model = {"active": False, "ranker": "v4", "reason": "Learning: 0 of 100 rows", "compare": {},
             "report": {"test_lists": 0, "ndcg": {"v1": 0.0, "v4": 0.0}}}
    assert "NDCG" not in ranker_text(model, "v4", {"source": "starter", "words": 8000})


def test_vectors_retrain_after_the_list_is_up(qtbot, rec, monkeypatch):
    order = []
    real = rec.refresh
    monkeypatch.setattr(rec, "refresh", lambda **kw: (real(**kw), order.append("refresh")))
    monkeypatch.setattr(rec, "retrain_vectors", lambda: order.append("retrain") or False)
    page = _page(qtbot, rec); page.open()
    qtbot.waitUntil(lambda: page.state == "list")
    qtbot.waitUntil(lambda: "retrain" in order)
    assert order == ["refresh", "retrain"]


def test_v7_v8_train_after_the_list_is_up(qtbot, rec, monkeypatch):
    """Review fix: v7/v8 training no longer holds the page on "Looking for new videos..."."""
    order, kwargs = [], []
    real = rec.refresh
    monkeypatch.setattr(rec, "refresh", lambda **kw: (kwargs.append(kw), real(**kw), order.append("refresh")))
    monkeypatch.setattr(rec, "train_experimental", lambda: order.append("experimental"))
    page = _page(qtbot, rec); page.open()
    qtbot.waitUntil(lambda: page.state == "list")
    qtbot.waitUntil(lambda: "experimental" in order)
    assert order[:2] == ["refresh", "experimental"] and kwargs == [{"experimental": False}]


# ---- v5: the Discovery block --------------------------------------------------------

from downloader.engine.recommend.bandit import Arm  # noqa: E402
from downloader.engine.recommend.bayes import fit_posterior  # noqa: E402
from downloader.ui.for_you_page import discovery_text  # noqa: E402


def _arms():
    return {"channel": Arm("channel", 1.0, 3.0), "newest": Arm("newest", 4.0, 8.25), "words": Arm("words", 0, 0)}


def test_discovery_text_random():
    info = {"mode": "random", "arms": _arms(), "pick": None, "posterior": None, "error": None}
    assert discovery_text(info).splitlines()[0] == "Discovery card: random from ranks 11–30"


def test_discovery_text_source_marks_the_winner():
    pick = {"picker": "source", "fallback": None, "draws": {"channel": 0.21, "newest": 0.38}}
    lines = discovery_text({"mode": "source", "arms": _arms(), "pick": pick, "posterior": None,
                            "error": None}).splitlines()
    assert lines[0] == "Discovery card: learns which search works"
    newest = next(x for x in lines if x.strip().startswith("newest"))
    assert "4.0 of 12.2 downloaded" in newest and "drew 0.38 ✓" in newest
    channel = next(x for x in lines if x.strip().startswith("channel"))
    assert "drew 0.21" in channel and "✓" not in channel
    words = next(x for x in lines if x.strip().startswith("words"))
    assert "drew" not in words  # no video of that kind in the pool this time
    assert lines[-1] == "Each refresh draws a guess per search kind; the highest guess picks the card."


def test_discovery_text_contextual_sorts_by_size():
    post = fit_posterior(["small", "big"], [[i % 2, i / 10] for i in range(20)],
                         [1 if i > 10 else 0 for i in range(20)], [1.0] * 20)
    lines = discovery_text({"mode": "contextual", "arms": _arms(), "pick": None, "posterior": post,
                            "error": None}).splitlines()
    assert lines[0] == "Discovery card: learns from every input (20 cards so far)"
    assert lines[1].strip().startswith("big") and lines[2].strip().startswith("small")
    assert "±" in lines[1]


def test_discovery_text_says_when_it_fell_back():
    pick = {"picker": "random", "fallback": "contextual", "draws": {}}
    text = discovery_text({"mode": "contextual", "arms": _arms(), "pick": pick, "posterior": None,
                           "error": None})
    assert text.splitlines()[-1] == "This time it fell back to random: no model yet."


def test_why_this_order_shows_the_discovery_block(qtbot, rec):
    page = _page(qtbot, rec)
    page.open()
    qtbot.waitUntil(lambda: bool(page.cards()), timeout=3000)
    page.findChild(QCheckBox, "why").setChecked(True)
    assert "Discovery card: learns which search works" in page.findChild(QLabel, "rankerInfo").text()


V7 = {"trained_at": 1.0, "vectors_at": None, "active": True, "reason": "v7 beats v4 on your newest lists (NDCG 0.62 vs 0.58)",
      "ndcg": {"v7": 0.62, "v4": 0.58}, "lists": 4, "model": "{}"}


def test_ranker_text_v7():
    from downloader.ui.for_you_page import ranker_text
    model = {"active": False, "reason": "Learning: 50 of 100 rows", "report": {}, "compare": {}}
    lines = ranker_text(model, "v7", None, V7).splitlines()
    assert lines[0] == "Ranked by: Two-tower (v7), learned from your lists"
    assert "v7 beats v4 on your newest lists (NDCG 0.62 vs 0.58)" in lines
    assert "v7 NDCG@10 0.62 vs v4 0.58" in lines


def test_ranker_text_v7_not_ranking_still_says_why():
    from downloader.ui.for_you_page import ranker_text
    info = {**V7, "active": False, "reason": "v7 needs Match by meaning", "ndcg": {}}
    text = ranker_text({"active": False, "reason": "x", "report": {}, "compare": {}}, "v1", None, info)
    assert text.splitlines()[0] == "Ranked by: Hand-tuned (v1)"
    assert "Two-tower (v7): v7 needs Match by meaning" in text


def test_ranker_text_v7_off():
    from downloader.ui.for_you_page import ranker_text
    text = ranker_text({"active": False, "reason": "x", "report": {}, "compare": {}}, "v1", None, None)
    assert "v7" not in text


def test_two_tower_card_text_is_a_score_not_a_chance():
    from downloader.engine.recommend.candidates import Candidate
    from downloader.engine.recommend.rank import Suggestion
    from downloader.ui.for_you_page import _score_text
    c = Candidate("v", "u", "t", "ch", 100, "q")
    text = _score_text(Suggestion(c, 3.4, 0.2, 0.1, None, [], two_tower=True))
    assert text.startswith("score 3.40 (two-tower)") and "chance" not in text and "%" not in text


V8 = {"trained_at": 1.0, "vectors_at": None, "active": True, "rival": "v7",
      "reason": "v8 beats v7 on your newest lists (NDCG 0.64 vs 0.61)", "ndcg": {"v8": 0.64, "v7": 0.61}, "lists": 4,
      "model": "{}"}


def test_ranker_text_v8():
    from downloader.ui.for_you_page import ranker_text
    model = {"active": False, "reason": "Learning: 50 of 100 rows", "report": {}, "compare": {}}
    lines = ranker_text(model, "v8", None, V7, sasrec=V8, leaned_on="Blank Space live").splitlines()
    assert lines[0] == "Ranked by: Sequence (v8), learned from your lists"
    assert "v8 beats v7 on your newest lists (NDCG 0.64 vs 0.61)" in lines
    assert "v8 NDCG@10 0.64 vs v7 0.61" in lines
    assert "Leaned most on: Blank Space live" in lines


def test_ranker_text_v8_not_ranking():
    from downloader.ui.for_you_page import ranker_text
    info = {**V8, "active": False, "reason": "v8 didn't beat v2 yet (NDCG 0.40 vs 0.55)",
            "ndcg": {"v8": 0.40, "v2": 0.55}}
    text = ranker_text({"active": False, "reason": "x", "report": {}, "compare": {}}, "v2", None, None,
                       sasrec=info, leaned_on=None)
    assert "Sequence (v8): v8 didn't beat v2 yet (NDCG 0.40 vs 0.55)" in text
    assert "Leaned most on" not in text


def test_ranker_text_v8_off():
    from downloader.ui.for_you_page import ranker_text
    text = ranker_text({"active": False, "reason": "x", "report": {}, "compare": {}}, "v1", None, None, sasrec=None)
    assert "v8" not in text


def test_why_text_asks_for_a_fresh_leaned_on(qtbot, rec, monkeypatch):
    """Review minor: v8 can start ranking after the list was drawn; the line comes from the recommender then."""
    page = _page(qtbot, rec)
    page.open()
    qtbot.waitUntil(lambda: bool(page.cards()), timeout=3000)
    monkeypatch.setattr(rec, "ranker", lambda: "v8")
    monkeypatch.setattr(rec, "sasrec_info", lambda: V8)
    monkeypatch.setattr(rec, "leaned_on", lambda: "Fresh piano title")
    page.findChild(QCheckBox, "why").setChecked(True)
    assert "Leaned most on: Fresh piano title" in page.findChild(QLabel, "rankerInfo").text()
