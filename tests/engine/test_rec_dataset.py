import json

from downloader.engine.recommend.dataset import IGNORED_WEIGHT, Row, rows

F = json.dumps({"similarity": 0.5, "affinity": 0.0, "shared": 1, "duration": 600, "rank": 1,
                "discovery": False, "kind": "words"})


class Log:
    """Builds rec_events tuples: (id, video_id, event, features, at, refresh)."""

    def __init__(self):
        self.events = []

    def shown(self, vid, refresh, features=F):
        self._add(vid, "shown", features, refresh)

    def act(self, vid, event):
        self._add(vid, event, None, None)

    def _add(self, vid, event, features, refresh):
        n = len(self.events) + 1
        self.events.append((n, vid, event, features, float(n), refresh))


def _labels(log):
    return [(r.video_id, r.label, r.weight, r.refresh) for r in rows(log.events)]


def test_download_dismiss_and_scrolled_past():
    log = Log()
    for v in ("a", "b", "c"):
        log.shown(v, 1.0)
    log.act("a", "downloaded")
    log.act("b", "dismissed")
    log.shown("x", 2.0)  # a newer refresh, so refresh 1's untouched rows are settled
    assert _labels(log) == [("a", 1, 1.0, 1.0), ("b", 0, 1.0, 1.0), ("c", 0, IGNORED_WEIGHT, 1.0)]


def test_the_newest_refreshs_untouched_rows_are_still_pending():
    log = Log()
    log.shown("a", 1.0)
    log.shown("b", 1.0)
    log.act("a", "downloaded")
    assert _labels(log) == [("a", 1, 1.0, 1.0)]  # b may still be downloaded


def test_an_action_belongs_to_the_last_showing_before_it():
    log = Log()
    log.shown("a", 1.0)
    log.shown("a", 2.0)
    log.act("a", "downloaded")
    log.shown("z", 3.0)
    assert _labels(log) == [("a", 0, IGNORED_WEIGHT, 1.0), ("a", 1, 1.0, 2.0)]


def test_a_showing_after_the_action_is_not_labelled_by_it():
    log = Log()
    log.shown("a", 1.0)
    log.act("a", "dismissed")
    log.shown("a", 2.0)  # shown again later (can't happen today, but must not reuse the dismissal)
    log.shown("z", 3.0)
    assert _labels(log)[:2] == [("a", 0, 1.0, 1.0), ("a", 0, IGNORED_WEIGHT, 2.0)]


def test_rows_from_before_1_2_1_are_left_out():
    log = Log()
    log.shown("old", None)
    log.act("old", "downloaded")
    log.shown("z", 2.0)
    assert _labels(log) == []


def test_only_the_newest_rows_are_kept():
    log = Log()
    for i in range(10):
        log.shown(f"v{i}", float(i))
    got = rows(log.events, limit=3)
    assert [r.video_id for r in got] == ["v6", "v7", "v8"]  # v9 is the newest refresh: pending


def test_a_row_carries_the_models_inputs():
    log = Log()
    log.shown("a", 1.0)
    log.act("a", "downloaded")
    r = rows(log.events)[0]
    assert isinstance(r, Row) and len(r.x) == 8 and r.x[0] == 0.5 and r.at == 1.0
