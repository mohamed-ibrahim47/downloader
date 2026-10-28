from PySide6.QtCore import QPoint, Qt

from downloader.ui.range_slider import RangeSlider


def _slider(qtbot, maximum=600, width=620):
    s = RangeSlider()
    qtbot.addWidget(s)
    s.set_maximum(maximum)
    s.resize(width, s.sizeHint().height())
    s.show()
    qtbot.waitExposed(s)
    return s


def test_starts_covering_everything(qtbot):
    s = _slider(qtbot)
    assert s.values() == (0, 600)


def test_set_values_clamps_and_orders(qtbot):
    s = _slider(qtbot)
    s.set_values(-5, 900)
    assert s.values() == (0, 600)
    s.set_values(300, 100)
    assert s.values() == (100, 300)


def test_values_never_meet(qtbot):
    s = _slider(qtbot)
    s.set_values(200, 200)
    start, end = s.values()
    assert end - start >= 1


def test_drag_left_handle(qtbot):
    s = _slider(qtbot)
    with qtbot.waitSignal(s.range_changed):
        qtbot.mousePress(s, Qt.LeftButton, pos=s.handle_pos(0))
        qtbot.mouseMove(s, s.pos_for(150))
        qtbot.mouseRelease(s, Qt.LeftButton, pos=s.pos_for(150))
    start, end = s.values()
    assert abs(start - 150) <= 2 and end == 600


def test_drag_right_handle(qtbot):
    s = _slider(qtbot)
    qtbot.mousePress(s, Qt.LeftButton, pos=s.handle_pos(1))
    qtbot.mouseMove(s, s.pos_for(450))
    qtbot.mouseRelease(s, Qt.LeftButton, pos=s.pos_for(450))
    start, end = s.values()
    assert start == 0 and abs(end - 450) <= 2


def test_left_handle_cannot_pass_right(qtbot):
    s = _slider(qtbot)
    s.set_values(100, 200)
    qtbot.mousePress(s, Qt.LeftButton, pos=s.handle_pos(0))
    qtbot.mouseMove(s, s.pos_for(500))
    qtbot.mouseRelease(s, Qt.LeftButton, pos=s.pos_for(500))
    start, end = s.values()
    assert start < end == 200


def test_click_on_track_moves_nearest_handle(qtbot):
    s = _slider(qtbot)
    qtbot.mouseClick(s, Qt.LeftButton, pos=s.pos_for(500))
    start, end = s.values()
    assert start == 0 and abs(end - 500) <= 2


def test_keyboard_moves_the_active_handle(qtbot):
    s = _slider(qtbot)
    s.set_values(100, 200)
    qtbot.mouseClick(s, Qt.LeftButton, pos=s.handle_pos(0))  # selects the left handle
    s.set_values(100, 200)
    qtbot.keyClick(s, Qt.Key_Right)
    assert s.values() == (101, 200)
    qtbot.keyClick(s, Qt.Key_Right, Qt.ShiftModifier)
    assert s.values() == (111, 200)
    qtbot.keyClick(s, Qt.Key_Left)
    assert s.values() == (110, 200)


def test_set_values_does_not_emit(qtbot):
    s = _slider(qtbot)
    seen = []
    s.range_changed.connect(lambda a, b: seen.append((a, b)))
    s.set_values(10, 20)
    assert seen == []


def test_accessible_name_describes_range(qtbot):
    s = _slider(qtbot)
    s.set_values(60, 150)
    assert s.accessibleDescription() == "from 1:00 to 2:30"
