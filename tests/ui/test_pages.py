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


def test_settings_page_learned_ranker_and_trainer(qtbot, monkeypatch):
    from PySide6.QtWidgets import QComboBox
    from downloader.engine.recommend import learn
    monkeypatch.setattr(learn, "_sklearn", lambda: object())  # pretend it's installed
    got = []
    page = SettingsPage(Settings(), got.append)
    qtbot.addWidget(page)
    page.findChild(QCheckBox, "rec_learned").setChecked(False)
    assert got[-1].rec_learned is False
    combo = page.findChild(QComboBox, "rec_trainer")
    combo.setCurrentIndex(combo.findData("sklearn"))
    assert got[-1].rec_trainer == "sklearn"


def test_trainer_choice_without_sklearn_is_greyed_out(qtbot, monkeypatch):
    from PySide6.QtWidgets import QComboBox
    from downloader.engine.recommend import learn
    monkeypatch.setattr(learn, "_sklearn", lambda: None)
    page = SettingsPage(Settings(), lambda s: None)
    qtbot.addWidget(page)
    combo = page.findChild(QComboBox, "rec_trainer")
    item = combo.model().item(combo.findData("sklearn"))
    assert not item.isEnabled() and "install scikit-learn" in combo.itemText(combo.findData("sklearn"))


def test_changing_the_trainer_retrains_in_the_background(win, qtbot, monkeypatch):
    trained = []
    monkeypatch.setattr(win.recommender, "train", lambda: trained.append(True))
    win.apply_settings(dataclasses.replace(win.settings, rec_trainer="sklearn"))
    qtbot.waitUntil(lambda: bool(trained), timeout=3000)


def test_settings_page_has_the_meaning_and_files_checkboxes(qtbot):
    got = []
    page = SettingsPage(Settings(), got.append)
    qtbot.addWidget(page)
    meaning = page.findChild(QCheckBox, "rec_meaning")
    files = page.findChild(QCheckBox, "rec_files")
    assert meaning.text() == "Match by meaning" and meaning.isChecked()
    assert files.text() == "Use file downloads for suggestions" and files.isChecked()
    meaning.setChecked(False)
    assert got[-1].rec_meaning is False
    files.setChecked(False)
    assert got[-1].rec_files is False
    page.set_settings(dataclasses.replace(got[-1], rec_meaning=True))
    assert meaning.isChecked()


@pytest.mark.parametrize("change", [{"rec_meaning": False}, {"rec_files": False}, {"rec_two_tower": True},
                                    {"rec_sasrec": True}])
def test_meaning_toggle_retrains(win, qtbot, monkeypatch, change):
    trained = []
    monkeypatch.setattr(win.recommender, "train", lambda: trained.append(True))
    win.apply_settings(dataclasses.replace(win.settings, **change))
    qtbot.waitUntil(lambda: bool(trained), timeout=3000)


def test_banner_turn_off_clears_the_setting(win):
    win.for_you_page.meaning_off_requested.emit()
    assert win.settings.rec_meaning is False


def test_settings_page_discovery_card_combo(qtbot):
    from PySide6.QtWidgets import QComboBox
    got = []
    page = SettingsPage(Settings(), got.append)
    qtbot.addWidget(page)
    combo = page.findChild(QComboBox, "rec_discovery")
    assert [combo.itemText(i) for i in range(combo.count())] == \
        ["Random", "Learn which search works", "Learn from every input"]
    assert combo.currentData() == "source"
    combo.setCurrentIndex(combo.findData("contextual"))
    assert got[-1].rec_discovery == "contextual"


def test_changing_the_discovery_mode_updates_why_this_order(win, monkeypatch):
    seen = []
    monkeypatch.setattr(win.for_you_page, "refresh_why", lambda: seen.append(True))
    win.apply_settings(dataclasses.replace(win.settings, rec_discovery="random"))
    assert seen


def test_two_tower_checkbox(qtbot):
    got = []
    page = SettingsPage(Settings(), got.append)
    qtbot.addWidget(page)
    box = page.findChild(QCheckBox, "rec_two_tower")
    assert box.text() == "Try the two-tower ranker (v7, experimental)" and not box.isChecked()
    boxes = [b.objectName() for b in page.findChildren(QCheckBox)]
    assert boxes.index("rec_two_tower") == boxes.index("rec_meaning") + 1
    box.setChecked(True)
    assert got[-1].rec_two_tower is True


def test_sasrec_checkbox(qtbot):
    got = []
    page = SettingsPage(Settings(), got.append)
    qtbot.addWidget(page)
    box = page.findChild(QCheckBox, "rec_sasrec")
    assert box.text() == "Try the sequence ranker (v8, experimental)" and not box.isChecked()
    boxes = [b.objectName() for b in page.findChildren(QCheckBox)]
    assert boxes.index("rec_sasrec") == boxes.index("rec_two_tower") + 1
    box.setChecked(True)
    assert got[-1].rec_sasrec is True
