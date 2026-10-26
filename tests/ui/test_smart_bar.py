import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton

from downloader.engine.errors import DownloadError
from downloader.engine.probe import FileInfo
from downloader.engine.settings import Settings
from downloader.engine.store import Store
from downloader.engine.youtube import PlaylistEntry, PlaylistInfo, VideoInfo
from downloader.ui import smart_bar
from downloader.ui.controller import Controller
from downloader.ui.smart_bar import SmartBar
from downloader.ui.theme import PALETTES

import json
from pathlib import Path

FORMATS = json.loads((Path(__file__).parents[1] / "engine" / "fixtures" / "yt_formats.json").read_text())


class IdleJob:
    def __init__(self, d, p, f):
        self.download = d

    def start(self):
        pass

    def pause(self):
        pass


@pytest.fixture
def bar(tmp_path, qtbot, monkeypatch):
    store = Store(tmp_path / "db.sqlite")
    ctl = Controller(store, Settings(folder_general=str(tmp_path)), job_factory=IdleJob)
    w = SmartBar(ctl, PALETTES["dark"])
    qtbot.addWidget(w)
    w.ctl = ctl
    yield w
    ctl.shutdown()
    store.close()


def _label(w, name):
    return w.findChild(QLabel, name).text()


def _button(w, text):
    return next(b for b in w.findChildren(QPushButton) if b.text() == text)


def test_file_link_shows_file_options(bar, qtbot, monkeypatch):
    monkeypatch.setattr(smart_bar, "probe", lambda url: FileInfo(url, "ubuntu.iso", 3_000_000_000, True))
    bar.set_text("https://releases.ubuntu.com/ubuntu.iso")
    qtbot.waitUntil(lambda: bar.state == "file", timeout=3000)
    assert _label(bar, "badge") == "File · resumable"
    assert _label(bar, "optTitle") == "ubuntu.iso"
    assert _label(bar, "sizeLabel") == "3.0 GB"
    assert _label(bar, "sizeNote") == "8 connections"
    assert not bar.findChild(QLabel, "qualityGroup") and not bar.quality_group.isVisible()


def test_video_link_quality_changes_size(bar, qtbot, monkeypatch):
    monkeypatch.setattr(smart_bar, "fetch_info",
                        lambda url: VideoInfo(url, "Big Buck Bunny", "Blender Foundation", 596, FORMATS))
    bar.set_text("https://www.youtube.com/watch?v=aqz-KE-bpKQ")
    qtbot.waitUntil(lambda: bar.state == "video", timeout=3000)
    bar.show()
    assert _label(bar, "badge") == "YouTube video"
    assert _label(bar, "sizeLabel") == "48.2 MB"
    assert bar.quality_group.isVisible()
    qtbot.mouseClick(_button(bar, "480p"), Qt.LeftButton)
    assert _label(bar, "sizeLabel") == "26.0 MB"
    qtbot.mouseClick(_button(bar, "Audio"), Qt.LeftButton)
    assert _label(bar, "sizeNote") == "M4A sound"


def test_playlist_shows_count_and_total(bar, qtbot, monkeypatch):
    entries = [PlaylistEntry(i, f"u{i}", f"Lecture {i}", 3000) for i in range(1, 13)]
    monkeypatch.setattr(smart_bar, "fetch_info", lambda url: PlaylistInfo(url, "Algorithms course", entries))
    bar.set_text("https://www.youtube.com/playlist?list=PL1")
    qtbot.waitUntil(lambda: bar.state == "playlist", timeout=3000)
    assert _label(bar, "badge") == "YouTube playlist · 12"
    assert _label(bar, "sizeNote") == "12 of 12 videos"
    assert _label(bar, "sizeLabel") == "5.0 GB"
    assert bar.action.text() == "Download all"


def test_invalid_text_shows_hint(bar, qtbot):
    bar.set_text("hello world")
    assert bar.state == "invalid"
    assert _label(bar, "hint") == "That doesn't look like a link"
    assert not bar.options.isVisibleTo(bar)


def test_download_adds_to_controller_and_clears(bar, qtbot, monkeypatch):
    monkeypatch.setattr(smart_bar, "probe", lambda url: FileInfo(url, "a.zip", 1000, True))
    bar.set_text("https://e.com/a.zip")
    qtbot.waitUntil(lambda: bar.state == "file", timeout=3000)
    qtbot.mouseClick(bar.action, Qt.LeftButton)
    assert [d.filename for d in bar.ctl.downloads()] == ["a.zip"]
    assert bar.findChild(QLineEdit, "smartInput").text() == ""
    assert bar.state == "empty"


def test_probe_error_shows_message(bar, qtbot, monkeypatch):
    def boom(url):
        raise DownloadError("server refused (403) · link may have expired")

    monkeypatch.setattr(smart_bar, "probe", boom)
    bar.set_text("https://e.com/a.zip")
    qtbot.waitUntil(lambda: bar.state == "error", timeout=3000)
    assert _label(bar, "hint") == "server refused (403) · link may have expired"


def test_stale_result_ignored(bar, qtbot, monkeypatch):
    import time

    def slow(url):
        time.sleep(0.3)
        return FileInfo(url, "old.zip", 1, True)

    monkeypatch.setattr(smart_bar, "probe", slow)
    bar.set_text("https://e.com/old.zip")
    bar.set_text("hello")
    qtbot.wait(600)
    assert bar.state == "invalid"
