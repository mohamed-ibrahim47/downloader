import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

from downloader.engine.models import Download, Kind, Status
from downloader.engine.probe import FileInfo
from downloader.engine.settings import Settings
from downloader.engine.store import Store
from downloader.ui import smart_bar
from downloader.ui.card import DownloadCard
from downloader.ui.controller import Controller
from downloader.ui.main_window import MainWindow


class IdleJob:
    def __init__(self, d, p, f):
        self.download = d

    def start(self):
        pass

    def pause(self):
        pass


def _d(status, n, **kw):
    return Download(url=f"https://e.com/{n}.zip", kind=Kind.FILE, filename=f"{n}.zip", folder="D:/x",
                    size=1_000_000, status=status, added_at=1000.0 + n, **kw)


@pytest.fixture
def win(tmp_path, qtbot, monkeypatch):
    monkeypatch.setattr(smart_bar, "probe", lambda url: FileInfo(url, "a.zip", 10, True))
    store = Store(tmp_path / "db.sqlite")
    store.add(_d(Status.QUEUED, 0))  # oldest waiting item: starts at launch
    store.add(_d(Status.QUEUED, 1))
    store.add(_d(Status.PAUSED, 2, downloaded=400_000))
    store.add(_d(Status.COMPLETED, 3, downloaded=1_000_000, finished_at=time.time()))
    settings = Settings(folder_general=str(tmp_path), max_active=1)
    ctl = Controller(store, settings, job_factory=IdleJob)
    w = MainWindow(ctl, settings, store)
    qtbot.addWidget(w)
    w.show()
    yield w
    ctl.shutdown()
    store.close()


def _chip(w, prefix):
    return next(b for b in w.findChildren(QPushButton, "chip") if b.text().startswith(prefix))


def _visible_cards(w):
    return [c for c in w.findChildren(DownloadCard) if c.isVisibleTo(w)]


def test_theme_toggle_switches_and_persists(win, qtbot):
    assert "#0E1015" in QApplication.instance().styleSheet()
    qtbot.mouseClick(win.findChild(QToolButton, "themeButton"), Qt.LeftButton)
    assert "#F4F5F7" in QApplication.instance().styleSheet()
    assert Settings.load(win.store).theme == "light"


def test_chips_show_counts(win):
    texts = [b.text() for b in win.findChildren(QPushButton, "chip")]
    assert texts == ["All 4", "Active 1", "Queued 1", "Completed 1"]


def test_all_hides_completed_behind_row(win):
    names = sorted(c.download.filename for c in _visible_cards(win))
    assert names == ["0.zip", "1.zip", "2.zip"]
    row = win.findChild(QPushButton, "completedRow")
    assert row.text().startswith("Completed today · 1 file · 1.0 MB")
    row.click()
    assert sorted(c.download.filename for c in _visible_cards(win)) == ["0.zip", "1.zip", "2.zip", "3.zip"]


def test_chips_filter_cards(win, qtbot):
    qtbot.mouseClick(_chip(win, "Queued"), Qt.LeftButton)
    assert [c.download.filename for c in _visible_cards(win)] == ["1.zip"]
    qtbot.mouseClick(_chip(win, "Completed"), Qt.LeftButton)
    assert [c.download.filename for c in _visible_cards(win)] == ["3.zip"]


def test_new_download_appears_as_card(win, qtbot):
    win.ctl.add_file(FileInfo("https://e.com/new.zip", "new.zip", 5, True))
    qtbot.waitUntil(lambda: "new.zip" in [c.download.filename for c in _visible_cards(win)], timeout=2000)
    assert _chip(win, "All").text() == "All 5"


def test_clipboard_link_fills_bar(win, qtbot):
    QGuiApplication.clipboard().setText("https://e.com/from-clipboard.zip")
    qtbot.waitUntil(lambda: win.smart_bar.input.text() == "https://e.com/from-clipboard.zip", timeout=2000)


def test_clipboard_ignored_when_off(win, qtbot):
    win.settings.watch_clipboard = False
    QGuiApplication.clipboard().setText("https://e.com/ignored.zip")
    qtbot.wait(300)
    assert win.smart_bar.input.text() == ""


def test_clipboard_ignores_plain_text(win, qtbot):
    QGuiApplication.clipboard().setText("just some words")
    qtbot.wait(300)
    assert win.smart_bar.input.text() == ""


def test_smoke_flag_exits_zero(tmp_path):
    # Its own process: main() owns and quits its QApplication, which must not be pytest's.
    import os
    import subprocess
    import sys
    env = {**os.environ, "DOWNLOADER_HOME": str(tmp_path / "home"), "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run([sys.executable, "-m", "downloader", "--smoke"], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "home" / "downloader.db").exists()
