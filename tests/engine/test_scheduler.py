from datetime import datetime

from downloader.engine.scheduler import Schedule, due_action

# 2026-10-26 is a Monday (weekday 0).
MON = datetime(2026, 10, 26)


def at(h, m, s=0, day=MON):
    return day.replace(hour=h, minute=m, second=s)


def test_due_start_when_crossed():
    s = Schedule(enabled=True, start="02:00")
    assert due_action(s, at(1, 59, 30), at(2, 0, 10)) == "start"


def test_not_due_before_start():
    s = Schedule(enabled=True, start="02:00")
    assert due_action(s, at(1, 58), at(1, 59)) is None


def test_not_due_on_disabled_day():
    s = Schedule(enabled=True, start="02:00", days=[1, 2, 3, 4, 5, 6])
    assert due_action(s, at(1, 59, 30), at(2, 0, 10)) is None


def test_stop_when_crossed():
    s = Schedule(enabled=True, start="02:00", stop="06:00")
    assert due_action(s, at(5, 59, 50), at(6, 0, 20)) == "stop"


def test_disabled_schedule_never_fires():
    s = Schedule(enabled=False, start="02:00")
    assert due_action(s, at(1, 59, 30), at(2, 0, 10)) is None


def test_crossing_midnight():
    s = Schedule(enabled=True, start="00:00")
    sunday_night = datetime(2026, 10, 25, 23, 59, 50)
    assert due_action(s, sunday_night, at(0, 0, 10)) == "start"
