from pathlib import Path

import pytest

from downloader.engine.models import Kind, Status
from downloader.engine.probe import FileInfo
from downloader.engine.settings import Settings
from downloader.engine.store import Store
from downloader.engine.youtube import FormatChoice, PlaylistEntry, PlaylistInfo, VideoInfo
from downloader.ui.controller import Controller


class IdleJob:
    def __init__(self, d, on_progress, on_finish):
        self.download = d
        self.on_progress = on_progress
        self.on_finish = on_finish

    def start(self):
        pass

    def pause(self):
        self.download.status = Status.PAUSED
        self.on_finish(self.download)


@pytest.fixture
def ctl(tmp_path, qapp):
    store = Store(tmp_path / "db.sqlite")
    settings = Settings(folder_general=str(tmp_path / "dl"), folder_video=str(tmp_path / "vid"),
                        folder_playlist=str(tmp_path / "pl"))
    jobs = []

    def factory(d, p, f):
        jobs.append(IdleJob(d, p, f))
        return jobs[-1]

    c = Controller(store, settings, job_factory=factory)
    c.jobs = jobs
    yield c
    c.shutdown()
    store.close()


def test_add_file_queues_and_emits_changed(ctl, qtbot):
    info = FileInfo("https://e.com/a.zip", "a.zip", 1000, True)
    with qtbot.waitSignal(ctl.changed) as sig:
        ctl.add_file(info)
    d = ctl.downloads()[0]
    assert sig.args == [d.id]
    assert d.folder == ctl.settings.folder_general
    assert d.connections == 8 and d.resumable and d.size == 1000
    assert d.status is Status.DOWNLOADING


def test_add_video_names_and_quality(ctl):
    info = VideoInfo("https://youtu.be/x", "Big Buck Bunny: 4K?", "Blender", 596, [])
    choice = FormatChoice("sel", 720, 48_000_000, "1080p not offered · got 720p", "mp4")
    ctl.add_video(info, choice)
    d = ctl.downloads()[0]
    assert d.kind is Kind.VIDEO
    assert d.filename == "Big Buck Bunny 4K.mp4"
    assert d.quality == "720p"
    assert d.note == "1080p not offered · got 720p"
    assert d.format_selector == "sel"
    assert d.title == "Big Buck Bunny: 4K?"


def test_add_video_avoids_existing_name(ctl):
    Path(ctl.settings.folder_video).mkdir(parents=True)
    (Path(ctl.settings.folder_video) / "Clip.mp4").write_bytes(b"x")
    ctl.add_video(VideoInfo("u", "Clip", "", None, []), FormatChoice("s", 480, None, None, "mp4"))
    assert ctl.downloads()[0].filename == "Clip (2).mp4"


def test_audio_choice_saves_m4a(ctl):
    ctl.add_video(VideoInfo("u", "Song", "", None, []), FormatChoice("s", None, None, None, "m4a"))
    d = ctl.downloads()[0]
    assert d.filename == "Song.m4a" and d.quality == "Audio"


def test_playlist_items_numbered_in_playlist_folder(ctl):
    info = PlaylistInfo("u", "Algorithms course", [
        PlaylistEntry(1, "u1", "Intro", 3120), PlaylistEntry(2, "u2", "Sorting", 2940),
        PlaylistEntry(3, "u3", "Heaps", 2820)])
    ctl.add_playlist(info, "720p", picked=[1, 3])
    items = sorted(ctl.downloads(), key=lambda d: d.playlist_index)
    assert [d.filename for d in items] == ["01 - Intro.mp4", "03 - Heaps.mp4"]
    assert all(d.folder.endswith("Algorithms course") for d in items)
    assert all(d.kind is Kind.PLAYLIST_ITEM and d.playlist_title == "Algorithms course" for d in items)
    assert items[0].size == round(3120 / 60 * 8.4 * 1_000_000)
    assert "height<=720" in items[0].format_selector


def test_playlist_without_numbers(ctl):
    ctl.settings.number_playlist_files = False
    ctl.add_playlist(PlaylistInfo("u", "P", [PlaylistEntry(1, "u1", "Intro", 60)]), "Audio", picked=[1])
    d = ctl.downloads()[0]
    assert d.filename == "Intro.m4a"


def test_progress_updates_speed(ctl, qtbot):
    ctl.add_file(FileInfo("https://e.com/a.zip", "a.zip", 10_000_000, True))
    job = ctl.jobs[0]
    for i in range(1, 6):
        job.download.downloaded = i * 1_000_000
        job.on_progress(job.download)
    qtbot.waitUntil(lambda: (ctl.speed_of(job.download.id)[0] or 0) > 0, timeout=2000)
    assert ctl.speed_of(job.download.id)[0] == pytest.approx(5_000_000 / 3)


def test_primary_on_downloading_pauses(ctl, qtbot):
    ctl.add_file(FileInfo("https://e.com/a.zip", "a.zip", 1000, True))
    d = ctl.downloads()[0]
    ctl.primary(d.id)
    qtbot.waitUntil(lambda: d.status is Status.PAUSED, timeout=2000)


def test_two_same_named_files_get_different_names(ctl):
    # Review C1: https://a/one/setup.zip and https://a/two/setup.zip queued together.
    ctl.add_file(FileInfo("https://a.com/one/setup.zip", "setup.zip", 10, True))
    ctl.add_file(FileInfo("https://a.com/two/setup.zip", "setup.zip", 10, True))
    assert sorted(d.filename for d in ctl.downloads()) == ["setup (2).zip", "setup.zip"]


def test_video_and_audio_of_same_title_dont_share_stem(ctl):
    ctl.add_video(VideoInfo("u", "Song", "", None, []), FormatChoice("s", 1080, None, None, "mp4"))
    ctl.add_video(VideoInfo("u", "Song", "", None, []), FormatChoice("s", None, None, None, "m4a"))
    assert sorted(d.filename for d in ctl.downloads()) == ["Song (2).m4a", "Song.mp4"]


def test_big_playlist_adds_quickly(ctl):
    # Review I6: 250 videos used to freeze the window for ~25 s.
    import time
    info = PlaylistInfo("u", "Big", [PlaylistEntry(i, f"u{i}", f"Video {i}", 60) for i in range(1, 251)])
    t0 = time.monotonic()
    ctl.add_playlist(info, "720p", picked=list(range(1, 251)))
    assert time.monotonic() - t0 < 3
    assert len(ctl.downloads()) == 250
    assert ctl.downloads()[-1].filename == "001 - Video 1.mp4"


def test_waiting_items_start_on_launch(tmp_path, qapp):
    # Review I2: items left waiting when the app closed start again when it opens.
    store = Store(tmp_path / "db.sqlite")
    from downloader.engine.models import Download
    store.add(Download(url="https://e.com/a", kind=Kind.FILE, filename="a.zip", folder=str(tmp_path)))
    c = Controller(store, Settings(folder_general=str(tmp_path)), job_factory=IdleJob)
    assert c.downloads()[0].status is Status.DOWNLOADING
    c.shutdown()
    store.close()


def test_apply_settings_sets_limit_and_concurrency(ctl):
    s = Settings(max_active=1, speed_limit_on=True, speed_limit_kbps=100)
    ctl.apply_settings(s)
    assert ctl.limiter.rate == 102_400
    assert ctl.queue._max_active == 1
