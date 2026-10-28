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


# ---- clips, checksums, already have it ----------------------------------------

import hashlib

from downloader.engine.checksum import Checksum
from downloader.engine.history import Existing

BUNNY = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"


def test_add_video_clip_name_size_and_id(ctl):
    d = ctl.add_video(VideoInfo(BUNNY, "Title", "", 596, []),
                      FormatChoice("best", 720, 48_200_000, None, "mp4"), clip=(750, 900))
    assert (d.filename, d.clip, d.video_id, d.size) == ("Title (12-30 to 15-00).mp4", (750, 900), "aqz-KE-bpKQ",
                                                         12_130_872)


def test_add_video_without_clip_keeps_full_size(ctl):
    d = ctl.add_video(VideoInfo(BUNNY, "Title", "", 596, []), FormatChoice("best", 720, 48_200_000, None, "mp4"))
    assert (d.filename, d.clip, d.size, d.video_id) == ("Title.mp4", None, 48_200_000, "aqz-KE-bpKQ")


def test_audio_clip_is_m4a(ctl):
    d = ctl.add_video(VideoInfo(BUNNY, "Title", "", 596, []),
                      FormatChoice("bestaudio", None, 3_400_000, None, "m4a"), clip=(0, 60))
    assert d.filename == "Title (0-00 to 1-00).m4a"


def test_add_file_stores_checksum(ctl):
    c = Checksum("sha256", hashlib.sha256(b"x").hexdigest())
    d = ctl.add_file(FileInfo("https://e.com/a.zip", "a.zip", 1, True), checksum=c, checksum_source="SHA256SUMS")
    assert d.checksum == str(c) and d.checksum_source == "SHA256SUMS"
    assert ctl.store.get(d.id).checksum == str(c)


def test_playlist_items_carry_video_id(ctl):
    info = PlaylistInfo("https://www.youtube.com/playlist?list=PL1", "PL",
                        [PlaylistEntry(1, "https://www.youtube.com/watch?v=AAAAAAAAAAA", "A", 60)])
    [d] = ctl.add_playlist(info, "720p", [1])
    assert d.video_id == "AAAAAAAAAAA"


def test_existing_and_already_have(ctl):
    d = ctl.add_file(FileInfo("https://e.com/a.zip", "a.zip", 1, True))
    assert ctl.existing("https://e.com/a.zip#x") == Existing(d, False)
    assert ctl.existing("https://e.com/b.zip") is None
    v = ctl.add_video(VideoInfo(BUNNY, "Bunny", "", 596, []), FormatChoice("best", 720, 1, None, "mp4"))
    v.status = Status.COMPLETED
    Path(v.folder).mkdir(parents=True, exist_ok=True)
    v.target_path.write_bytes(b"v")
    info = PlaylistInfo("u", "PL", [PlaylistEntry(1, BUNNY, "Bunny", 596),
                                    PlaylistEntry(2, "https://www.youtube.com/watch?v=BBBBBBBBBBB", "B", 9)])
    assert ctl.already_have(info) == {1}


def test_keep_both_clears_note(ctl, qtbot):
    d = ctl.add_file(FileInfo("https://e.com/a.zip", "a.zip", 1, True))
    d.duplicate_of = "b.zip"
    with qtbot.waitSignal(ctl.changed):
        ctl.keep_both(d.id)
    assert d.duplicate_of is None and ctl.store.get(d.id).duplicate_of is None


def test_delete_copy_removes_and_deletes(ctl, qtbot):
    d = ctl.add_file(FileInfo("https://e.com/a.zip", "a.zip", 1, True))
    ctl.queue.pause(d.id)
    d.status, d.duplicate_of = Status.COMPLETED, "b.zip"
    Path(d.folder).mkdir(parents=True, exist_ok=True)
    d.target_path.write_bytes(b"x")
    (Path(d.folder) / "b.zip").write_bytes(b"x")  # still identical when deleting
    with qtbot.waitSignal(ctl.removed, timeout=3000):
        ctl.delete_copy(d.id)
    assert not d.target_path.exists() and ctl.get(d.id) is None


def test_redownload_runs_in_background(ctl, qtbot):
    d = ctl.add_file(FileInfo("https://e.com/a.zip", "a.zip", 1, True))
    ctl.queue.pause(d.id)
    d.status = Status.FAILED
    ctl.redownload(d.id)
    qtbot.waitUntil(lambda: d.status in (Status.QUEUED, Status.DOWNLOADING), timeout=3000)


def _copy_pair(ctl, original: bytes):
    d = ctl.add_file(FileInfo("https://e.com/setup.zip", "setup (2).zip", 5, True))
    ctl.queue.pause(d.id)
    d.status, d.duplicate_of = Status.COMPLETED, "setup.zip"
    Path(d.folder).mkdir(parents=True, exist_ok=True)
    d.target_path.write_bytes(b"12345")
    if original is not None:
        (Path(d.folder) / "setup.zip").write_bytes(original)
    return d


def test_delete_copy_keeps_file_when_original_is_gone(ctl, qtbot):
    d = _copy_pair(ctl, None)
    ctl.delete_copy(d.id)
    qtbot.waitUntil(lambda: d.duplicate_of is None, timeout=3000)
    assert d.target_path.exists() and d.duplicate_of is None and ctl.get(d.id) is not None


def test_delete_copy_keeps_file_when_original_changed(ctl, qtbot):
    d = _copy_pair(ctl, b"54321")  # same size, different bytes now
    ctl.delete_copy(d.id)
    qtbot.waitUntil(lambda: d.duplicate_of is None, timeout=3000)
    assert d.target_path.exists() and d.duplicate_of is None


def test_delete_copy_deletes_when_still_identical(ctl, qtbot):
    d = _copy_pair(ctl, b"12345")
    with qtbot.waitSignal(ctl.removed, timeout=3000):
        ctl.delete_copy(d.id)
    assert not d.target_path.exists() and (Path(d.folder) / "setup.zip").exists()


def test_whole_video_clip_is_saved_as_the_video(ctl):
    d = ctl.add_video(VideoInfo(BUNNY, "Title", "", 596, []),
                      FormatChoice("best", 720, 48_200_000, None, "mp4"), clip=(0, 596))
    assert (d.filename, d.clip, d.size) == ("Title.mp4", None, 48_200_000)
