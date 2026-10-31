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


# ---- clips -------------------------------------------------------------------------

def _video(bar, qtbot, monkeypatch, url="https://www.youtube.com/watch?v=aqz-KE-bpKQ"):
    monkeypatch.setattr(smart_bar, "fetch_info",
                        lambda u: VideoInfo(u, "Big Buck Bunny", "Blender Foundation", 596, FORMATS))
    bar.set_text(url)
    qtbot.waitUntil(lambda: bar.state == "video", timeout=3000)
    bar.show()


def test_smart_bar_clip_size_and_download(bar, qtbot, monkeypatch):
    _video(bar, qtbot, monkeypatch)
    assert bar.clip.toggle.isVisible()
    bar.clip.toggle.setChecked(True)
    bar.clip.from_edit.setText("0:00")
    bar.clip.to_edit.setText("4:58")
    assert _label(bar, "sizeLabel") == "≈ 24.1 MB"
    assert _label(bar, "sizeNote") == "clip · estimate"
    qtbot.mouseClick(bar.action, Qt.LeftButton)
    d = bar.ctl.downloads()[0]
    assert d.clip == (0, 298) and d.filename == "Big Buck Bunny (0-00 to 4-58).mp4"


def test_smart_bar_invalid_clip_disables_download(bar, qtbot, monkeypatch):
    _video(bar, qtbot, monkeypatch)
    bar.clip.toggle.setChecked(True)
    bar.clip.to_edit.setText("20:00")
    assert not bar.action.isEnabled()
    bar.clip.to_edit.setText("2:00")
    assert bar.action.isEnabled()


def test_clip_switch_hidden_for_files(bar, qtbot, monkeypatch):
    monkeypatch.setattr(smart_bar, "probe", lambda url: FileInfo(url, "a.zip", 10, True))
    bar.set_text("https://e.com/a.zip")
    qtbot.waitUntil(lambda: bar.state == "file", timeout=3000)
    bar.show()
    assert not bar.clip.toggle.isVisible() and not bar.clip.isVisible()


def test_new_link_resets_clip(bar, qtbot, monkeypatch):
    _video(bar, qtbot, monkeypatch)
    bar.clip.toggle.setChecked(True)
    _video(bar, qtbot, monkeypatch, url="https://youtu.be/aqz-KE-bpKQ?t=90")
    assert not bar.clip.on() and bar.clip.from_edit.text() == "1:30"


def test_format_dialog_clip(qtbot):
    from downloader.ui.dialogs import FormatDialog
    info = VideoInfo("https://www.youtube.com/watch?v=aqz-KE-bpKQ", "Bunny", "", 596, FORMATS)
    dialog = FormatDialog(info, "D:/v", "720p", False, clip_url=info.url + "&t=60")
    qtbot.addWidget(dialog)
    assert dialog.clip() is None
    dialog.clip_fields.toggle.setChecked(True)
    dialog.clip_fields.to_edit.setText("2:30")
    assert dialog.clip() == (60, 150)


# ---- you already have this ---------------------------------------------------------

import datetime

from downloader.engine.models import Download, Kind, Status

BUNNY = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"


def _have(bar, tmp_path, url, name, kind=Kind.FILE, status=Status.COMPLETED, on_disk=True, **kw):
    d = Download(url=url, kind=kind, filename=name, folder=str(tmp_path), **kw)
    bar.ctl.queue.add(d)
    d.status = status
    if status is Status.COMPLETED:
        d.finished_at = datetime.datetime(datetime.date.today().year, 10, 26, 9).timestamp()
    if on_disk:
        (tmp_path / name).write_bytes(b"x")
    return d


def _file(bar, qtbot, monkeypatch, url="https://e.com/a.zip"):
    monkeypatch.setattr(smart_bar, "probe", lambda u: FileInfo(u, "a.zip", 10, True))
    bar.set_text(url)
    qtbot.waitUntil(lambda: bar.state == "file", timeout=3000)
    bar.show()


def _notice(bar):
    label = bar.findChild(QLabel, "notice")
    return label.text() if label.isVisible() else None


def test_notice_for_downloaded_file(bar, qtbot, monkeypatch, tmp_path):
    d = _have(bar, tmp_path, "https://e.com/a.zip", "a.zip")
    opened = []
    monkeypatch.setattr(bar.ctl, "open_folder", opened.append)
    _file(bar, qtbot, monkeypatch, "https://e.com/a.zip#top")
    assert _notice(bar) == "You downloaded this on 26 Oct"
    assert bar.action.isEnabled()
    qtbot.mouseClick(_button(bar, "Open folder"), Qt.LeftButton)
    assert opened == [d]


def test_no_notice_for_new_link(bar, qtbot, monkeypatch, tmp_path):
    _file(bar, qtbot, monkeypatch)
    assert _notice(bar) is None


def test_no_notice_when_file_deleted(bar, qtbot, monkeypatch, tmp_path):
    _have(bar, tmp_path, "https://e.com/a.zip", "a.zip", on_disk=False)
    _file(bar, qtbot, monkeypatch)
    assert _notice(bar) is None


def test_download_again_adds_second_copy(bar, qtbot, monkeypatch, tmp_path):
    _have(bar, tmp_path, "https://e.com/a.zip", "a.zip")
    _file(bar, qtbot, monkeypatch)
    qtbot.mouseClick(_button(bar, "Download again"), Qt.LeftButton)
    newest = bar.ctl.downloads()[0]
    assert newest.filename == "a (2).zip" and bar.state == "empty"


def test_notice_for_same_video_other_link(bar, qtbot, monkeypatch, tmp_path):
    _have(bar, tmp_path, BUNNY, "Bunny.mp4", kind=Kind.VIDEO, video_id="aqz-KE-bpKQ")
    _video(bar, qtbot, monkeypatch, url="https://youtu.be/aqz-KE-bpKQ?t=5")
    assert _notice(bar) == "You downloaded this on 26 Oct"


def test_notice_clip_vs_full(bar, qtbot, monkeypatch, tmp_path):
    _have(bar, tmp_path, BUNNY, "Bunny (1-00 to 2-00).mp4", kind=Kind.VIDEO, video_id="aqz-KE-bpKQ",
          clip_start=60, clip_end=120)
    _video(bar, qtbot, monkeypatch)
    assert _notice(bar) == "You downloaded a clip of this (1:00–2:00) on 26 Oct"
    bar.clip.toggle.setChecked(True)
    bar.clip.from_edit.setText("1:00")
    bar.clip.to_edit.setText("2:00")
    assert _notice(bar) == "You downloaded this on 26 Oct"


def test_notice_whole_video_when_picking_a_clip(bar, qtbot, monkeypatch, tmp_path):
    _have(bar, tmp_path, BUNNY, "Bunny.mp4", kind=Kind.VIDEO, video_id="aqz-KE-bpKQ")
    _video(bar, qtbot, monkeypatch)
    bar.clip.toggle.setChecked(True)
    bar.clip.to_edit.setText("2:00")
    assert _notice(bar) == "You downloaded the whole video on 26 Oct"


def test_notice_for_file_already_in_folder_without_history(bar, qtbot, monkeypatch, tmp_path):
    (tmp_path / "a.zip").write_bytes(b"x")  # removed from the list, but still on disk
    opened = []
    monkeypatch.setattr(bar.ctl, "reveal", opened.append)
    _file(bar, qtbot, monkeypatch)
    assert _notice(bar) == "a.zip is already in the folder"
    qtbot.mouseClick(_button(bar, "Open folder"), Qt.LeftButton)
    assert opened == [tmp_path / "a.zip"]
    assert _button(bar, "Download again").isVisible()


def test_notice_for_video_already_in_folder_without_history(bar, qtbot, monkeypatch, tmp_path):
    bar.ctl.settings.folder_video = str(tmp_path)
    (tmp_path / "Big Buck Bunny.mp4").write_bytes(b"x")
    _video(bar, qtbot, monkeypatch)
    assert _notice(bar) == "Big Buck Bunny.mp4 is already in the folder"


def test_whole_range_clip_is_the_full_video(bar, qtbot, monkeypatch):
    _video(bar, qtbot, monkeypatch)
    bar.clip.toggle.setChecked(True)  # the bar starts covering the whole video
    assert _label(bar, "sizeLabel") == "48.2 MB"
    qtbot.mouseClick(bar.action, Qt.LeftButton)
    d = bar.ctl.downloads()[0]
    assert d.clip is None and d.filename == "Big Buck Bunny.mp4"


def test_show_it_emits(bar, qtbot, monkeypatch, tmp_path):
    d = _have(bar, tmp_path, "https://e.com/a.zip", "a.zip", status=Status.QUEUED, on_disk=False)
    _file(bar, qtbot, monkeypatch)
    assert _notice(bar) == "Already in your list"
    with qtbot.waitSignal(bar.show_requested) as sig:
        qtbot.mouseClick(_button(bar, "Show it"), Qt.LeftButton)
    assert sig.args == [d.id]


def test_playlist_unticks_what_you_have(bar, qtbot, monkeypatch, tmp_path):
    _have(bar, tmp_path, "https://www.youtube.com/watch?v=BBBBBBBBBBB", "B.mp4", kind=Kind.VIDEO,
          video_id="BBBBBBBBBBB")
    entries = [PlaylistEntry(i, f"https://www.youtube.com/watch?v={c * 11}", f"V{i}", 60)
               for i, c in ((1, "A"), (2, "B"), (3, "C"))]
    monkeypatch.setattr(smart_bar, "fetch_info", lambda u: PlaylistInfo(u, "PL", entries))
    bar.set_text("https://www.youtube.com/playlist?list=PL1")
    qtbot.waitUntil(lambda: bar.state == "playlist", timeout=3000)
    assert "1 already downloaded" in _label(bar, "optMeta")
    assert bar.action.text() == "Download 2 videos"


def test_playlist_dialog_marks_downloaded(qtbot):
    from downloader.ui.dialogs import PlaylistDialog
    entries = [PlaylistEntry(1, "u1", "One", 60), PlaylistEntry(2, "u2", "Two", 60)]
    dialog = PlaylistDialog(PlaylistInfo("u", "PL", entries), {1}, have={2})
    qtbot.addWidget(dialog)
    texts = [dialog._list.item(i).text() for i in range(2)]
    assert "already downloaded" not in texts[0] and texts[1].endswith("already downloaded")


# ---- checksums ---------------------------------------------------------------------

import hashlib

from PySide6.QtWidgets import QLineEdit

from downloader.engine.checksum import Checksum

H = hashlib.sha256(b"x").hexdigest()


def _sum_field(bar):
    return bar.findChild(QLineEdit, "checksumInput")


def _sum_note(bar):
    label = bar.findChild(QLabel, "checksumNote")
    return label.text() if label.isVisible() else ""


def test_checksum_field_shown_for_files(bar, qtbot, monkeypatch):
    _file(bar, qtbot, monkeypatch)
    assert _sum_field(bar).isVisible()
    assert "optional" in _sum_field(bar).placeholderText().lower()


def test_checksum_field_only_for_files(bar, qtbot, monkeypatch):
    _video(bar, qtbot, monkeypatch)
    assert not _sum_field(bar).isVisible()


def test_checksum_found_automatically(bar, qtbot, monkeypatch):
    seen = []
    monkeypatch.setattr(smart_bar, "lookup_sums",
                        lambda url, name: seen.append((url, name)) or (Checksum("sha256", H), "SHA256SUMS"))
    _file(bar, qtbot, monkeypatch)
    qtbot.waitUntil(lambda: _sum_note(bar) == "checksum found (SHA256SUMS)", timeout=3000)
    assert seen == [("https://e.com/a.zip", "a.zip")]
    qtbot.mouseClick(bar.action, Qt.LeftButton)
    d = bar.ctl.downloads()[0]
    assert d.checksum == f"sha256:{H}" and d.checksum_source == "SHA256SUMS"


def test_pasted_checksum_wins(bar, qtbot, monkeypatch):
    monkeypatch.setattr(smart_bar, "lookup_sums", lambda url, name: (Checksum("sha256", "c" * 64), "SHA256SUMS"))
    _file(bar, qtbot, monkeypatch)
    qtbot.waitUntil(lambda: "found" in _sum_note(bar), timeout=3000)
    _sum_field(bar).setText(f"{H.upper()}  a.zip")
    assert _sum_note(bar) == "SHA-256 · checked after download"
    qtbot.mouseClick(bar.action, Qt.LeftButton)
    d = bar.ctl.downloads()[0]
    assert d.checksum == f"sha256:{H}" and d.checksum_source == "pasted"


def test_invalid_checksum_disables_download(bar, qtbot, monkeypatch):
    _file(bar, qtbot, monkeypatch)
    _sum_field(bar).setText("not a hash")
    assert _sum_note(bar) == "paste a checksum: 32, 40, 64 or 128 hex characters"
    assert not bar.action.isEnabled()
    _sum_field(bar).clear()
    assert bar.action.isEnabled() and _sum_note(bar) == ""


def test_stale_lookup_ignored(bar, qtbot, monkeypatch):
    import threading
    gate = threading.Event()

    def slow(url, name):
        gate.wait(2)
        return Checksum("sha256", H), "SHA256SUMS"

    monkeypatch.setattr(smart_bar, "lookup_sums", slow)
    _file(bar, qtbot, monkeypatch)
    monkeypatch.setattr(smart_bar, "lookup_sums", lambda url, name: None)
    _file(bar, qtbot, monkeypatch, url="https://e.com/b.zip")
    gate.set()
    qtbot.wait(300)
    assert _sum_note(bar) == ""


def test_lookup_failure_is_silent(bar, qtbot, monkeypatch):
    _file(bar, qtbot, monkeypatch)
    qtbot.wait(100)
    assert _sum_note(bar) == "" and bar.hint.text() != ""
    assert bar.state == "file"


def test_new_link_clears_pasted_checksum(bar, qtbot, monkeypatch):
    _file(bar, qtbot, monkeypatch)
    _sum_field(bar).setText(H)
    _file(bar, qtbot, monkeypatch, url="https://e.com/b.zip")
    assert _sum_field(bar).text() == ""


def test_download_again_respects_invalid_checksum(bar, qtbot, monkeypatch, tmp_path):
    _have(bar, tmp_path, "https://e.com/a.zip", "a.zip")
    _file(bar, qtbot, monkeypatch)
    _sum_field(bar).setText("not a hash")
    before = len(bar.ctl.downloads())
    assert not _button(bar, "Download again").isEnabled()
    bar._download()  # even if called directly, nothing unchecked is queued
    assert len(bar.ctl.downloads()) == before


def test_download_again_respects_invalid_clip(bar, qtbot, monkeypatch, tmp_path):
    _have(bar, tmp_path, BUNNY, "Bunny.mp4", kind=Kind.VIDEO, video_id="aqz-KE-bpKQ")
    _video(bar, qtbot, monkeypatch)
    bar.clip.toggle.setChecked(True)
    bar.clip.to_edit.setText("1:00")
    bar.clip.from_edit.setText("2:00")  # From after To; the full-video notice may show meanwhile
    before = len(bar.ctl.downloads())
    bar._download()
    assert len(bar.ctl.downloads()) == before


def test_plain_http_link_is_not_looked_up(bar, qtbot, monkeypatch):
    seen = []
    monkeypatch.setattr(smart_bar, "lookup_sums", lambda url, name: seen.append(url))
    _file(bar, qtbot, monkeypatch, url="http://e.com/a.zip")
    qtbot.wait(100)
    assert seen == []
    assert _sum_note(bar) == "http link, so no automatic check · paste a checksum from the site"


def test_https_link_shows_no_http_note(bar, qtbot, monkeypatch):
    _file(bar, qtbot, monkeypatch)
    assert _sum_note(bar) == ""


def test_pasted_weak_checksum_is_labelled(bar, qtbot, monkeypatch):
    _file(bar, qtbot, monkeypatch)
    _sum_field(bar).setText("a" * 32)
    assert _sum_note(bar) == "MD5 · weak check · checked after download"
