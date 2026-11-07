import dataclasses
from datetime import datetime

import pytest
from PySide6.QtCore import QTime
from PySide6.QtWidgets import QCheckBox, QLineEdit, QPushButton, QTimeEdit

from downloader.engine.models import Download, Kind, Status
from downloader.engine.recommend.service import Recommender
from downloader.engine.scheduler import Schedule
from downloader.engine.settings import Settings
from downloader.engine.store import Store
from downloader.ui.controller import Controller
from downloader.ui.main_window import NAV, MainWindow
from downloader.ui.pages import SchedulerPage, SettingsPage


class IdleJob:
    def __init__(self, d, p, f):
        self.download = d

    def start(self):
        pass

    def pause(self):
        pass


class FakeTray:
    def __init__(self):
        self.messages = []

    def notify(self, title, text):
        self.messages.append((title, text))


def test_settings_page_saves_folder(qtbot):
    got = []
    page = SettingsPage(Settings(folder_video="D:/old"), got.append)
    qtbot.addWidget(page)
    field = page.findChild(QLineEdit, "folder_video")
    field.setText("D:/new videos")
    field.editingFinished.emit()
    assert got[-1].folder_video == "D:/new videos"


def test_settings_page_toggles(qtbot):
    got = []
    page = SettingsPage(Settings(), got.append)
    qtbot.addWidget(page)
    page.findChild(QCheckBox, "subtitles").setChecked(True)
    assert got[-1].subtitles is True
    page.findChild(QCheckBox, "number_playlist_files").setChecked(False)
    assert got[-1].number_playlist_files is False


def test_scheduler_page_round_trip(qtbot):
    got = []
    s = Settings(schedule=Schedule(enabled=True, start="03:30", days=[0, 2]))
    page = SchedulerPage(s, got.append)
    qtbot.addWidget(page)
    assert page.findChild(QTimeEdit, "start").time() == QTime(3, 30)
    page.findChild(QCheckBox, "use_stop").setChecked(True)
    page.findChild(QTimeEdit, "stop").setTime(QTime(7, 15))
    friday = next(b for b in page.findChildren(QPushButton, "segButton") if b.text() == "Fri")
    friday.click()
    sched = got[-1].schedule
    assert (sched.enabled, sched.start, sched.stop) == (True, "03:30", "07:15")
    assert sched.days == [0, 2, 4]


@pytest.fixture
def win(tmp_path, qtbot):
    store = Store(tmp_path / "db.sqlite")
    store.add(Download(url="https://e.com/a", kind=Kind.FILE, filename="a.zip", folder=str(tmp_path),
                       status=Status.PAUSED))
    settings = Settings(folder_general=str(tmp_path))
    ctl = Controller(store, settings, job_factory=IdleJob)
    w = MainWindow(ctl, settings, store, recommender=Recommender(store, search=lambda q, l: []))
    w.tray = FakeTray()
    qtbot.addWidget(w)
    yield w
    ctl.shutdown()
    store.close()


def test_for_you_sits_under_downloads_in_the_rail():
    assert [key for key, _icon, _label in NAV][:2] == ["downloads", "for_you"]
    assert dict((k, (i, l)) for k, i, l in NAV)["for_you"] == ("spark", "For you")


def test_for_you_rail_opens_page(win):
    win._nav_buttons["for_you"].click()
    assert win.pages.currentWidget() is win.for_you_page


def test_leaving_for_you_tells_the_page(win):
    win._nav_buttons["for_you"].click()
    assert win.for_you_page._active
    win._nav_buttons["downloads"].click()
    assert not win.for_you_page._active


def test_download_from_for_you_fills_smart_bar(win, monkeypatch):
    seen = []
    monkeypatch.setattr(win.smart_bar, "set_text", seen.append)
    win._nav_buttons["for_you"].click()
    win.for_you_page.download_requested.emit("https://www.youtube.com/watch?v=abc")
    assert seen == ["https://www.youtube.com/watch?v=abc"] and win.pages.currentIndex() == 0
    assert win._nav_buttons["downloads"].isChecked()


def test_turning_suggestions_off_hides_the_page(win):
    win._nav_buttons["for_you"].click()
    win.apply_settings(dataclasses.replace(win.settings, suggest_videos=False))
    assert win._nav_buttons["for_you"].isHidden() and win.pages.currentIndex() == 0
    assert win._nav_buttons["downloads"].isChecked()


def test_suggestions_off_at_start_hides_the_rail_button(tmp_path, qtbot):
    store = Store(tmp_path / "db.sqlite")
    settings = Settings(folder_general=str(tmp_path), suggest_videos=False)
    ctl = Controller(store, settings, job_factory=IdleJob)
    w = MainWindow(ctl, settings, store, recommender=Recommender(store, search=lambda q, l: []))
    qtbot.addWidget(w)
    assert w._nav_buttons["for_you"].isHidden()
    ctl.shutdown()
    store.close()


def test_settings_page_suggest_toggle(qtbot):
    got = []
    page = SettingsPage(Settings(), got.append)
    qtbot.addWidget(page)
    box = page.findChild(QCheckBox, "suggest_videos")
    assert box.text() == "Suggest videos on the For you page" and box.isChecked()
    box.setChecked(False)
    assert got[-1].suggest_videos is False


def test_queue_finished_notify_calls_tray(win, qtbot):
    win.apply_settings(Settings(folder_general=win.settings.folder_general, when_done="notify"))
    win.ctl.queue_finished.emit()
    assert win.tray.messages == [("All downloads finished", "Your queue is empty.")]


def test_queue_finished_sleep_puts_pc_to_sleep(win):
    slept = []
    win.sleep_pc = lambda: slept.append(True)
    win.apply_settings(Settings(folder_general=win.settings.folder_general, when_done="sleep"))
    win.ctl.queue_finished.emit()
    assert slept == [True]


def test_queue_finished_nothing_is_quiet(win):
    win.ctl.queue_finished.emit()
    assert win.tray.messages == []


def test_download_failure_notifies(win):
    d = win.ctl.downloads()[0]
    d.status, d.error = Status.FAILED, "disk is full"
    win._on_finished(d.id)
    assert win.tray.messages == [("a.zip failed", "disk is full")]


def test_scheduler_timer_starts_queue(win):
    started = []
    win.ctl.start_queue = lambda: started.append("start")
    win.apply_settings(Settings(folder_general=win.settings.folder_general,
                                schedule=Schedule(enabled=True, start="02:00")))
    win._sched_prev = datetime(2026, 10, 26, 1, 59, 30)
    win._scheduler_tick(datetime(2026, 10, 26, 2, 0, 10))
    assert started == ["start"]
    assert win._sched_prev == datetime(2026, 10, 26, 2, 0, 10)


def test_rail_opens_pages(win, qtbot):
    win._nav_buttons["settings"].click()
    assert win.pages.currentWidget() is win.settings_page
    win._nav_buttons["scheduler"].click()
    assert win.pages.currentWidget() is win.scheduler_page
