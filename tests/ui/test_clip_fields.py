from downloader.ui.clip_fields import ClipFields

URL = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"


def _fields(qtbot, duration=596, url=URL):
    w = ClipFields()
    qtbot.addWidget(w)
    w.set_video(duration, url)
    return w


def _set(w, start, end):
    w.toggle.setChecked(True)
    w.from_edit.setText(start)
    w.to_edit.setText(end)


def test_prefills_from_link(qtbot):
    w = _fields(qtbot, url=URL + "&t=12m30s")
    assert (w.from_edit.text(), w.to_edit.text(), w.on()) == ("12:30", "9:56", False)
    assert w.clip() is None


def test_no_start_in_link_leaves_from_empty_meaning_start(qtbot):
    w = _fields(qtbot)
    w.toggle.setChecked(True)
    w.to_edit.setText("1:00")
    assert w.from_edit.text() == "" and w.clip() == (0, 60)


def test_clip_range(qtbot):
    w = _fields(qtbot)
    _set(w, "1:00", "2:30")
    assert w.clip() == (60, 150) and w.problem() is None


def test_clip_from_after_to(qtbot):
    w = _fields(qtbot)
    _set(w, "3:00", "2:30")
    assert w.problem() == "From must be before To" and w.clip() is None


def test_clip_to_past_end(qtbot):
    w = _fields(qtbot)
    _set(w, "1:00", "10:00")
    assert w.problem() == "the video is 9 min 56 s long" and w.clip() is None


def test_clip_needs_to_when_length_unknown(qtbot):
    w = _fields(qtbot, duration=None)
    w.toggle.setChecked(True)
    assert w.to_edit.text() == ""
    assert w.problem() == "enter where the clip ends" and w.clip() is None


def test_bad_time_text(qtbot):
    w = _fields(qtbot)
    _set(w, "abc", "2:00")
    assert w.problem() == "use a time like 12:30 or 1:02:03"


def test_off_means_no_problem(qtbot):
    w = _fields(qtbot)
    w.from_edit.setText("abc")
    assert not w.on() and w.problem() is None and w.clip() is None


def test_changed_signal(qtbot):
    w = _fields(qtbot)
    with qtbot.waitSignal(w.changed):
        w.toggle.setChecked(True)
    with qtbot.waitSignal(w.changed):
        w.to_edit.setText("1:00")


def test_problem_shown_as_plain_text(qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel
    w = _fields(qtbot)
    _set(w, "3:00", "2:30")
    label = w.findChild(QLabel, "clipProblem")
    assert label.text() == "From must be before To" and label.textFormat() == Qt.PlainText
