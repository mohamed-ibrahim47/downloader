import time

import pytest

from downloader.engine.models import Download, Kind, Status
from downloader.engine.queue import DownloadQueue
from downloader.engine.store import Store


class FakeJob:
    def __init__(self, download, on_progress, on_finish):
        self.download = download
        self._on_finish = on_finish
        self.started = False

    def start(self):
        self.started = True

    def pause(self):
        self.finish(Status.PAUSED)

    def finish(self, status, error=None):
        self.download.status = status
        self.download.error = error
        self._on_finish(self.download)


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    jobs = {}

    def factory(d, on_progress, on_finish):
        jobs[d.id] = FakeJob(d, on_progress, on_finish)
        return jobs[d.id]

    q = DownloadQueue(store, factory, max_active=3)
    events = {"changed": [], "removed": [], "finished": 0}
    q.on_change = lambda d: events["changed"].append((d.id, d.status))
    q.on_removed = lambda i: events["removed"].append(i)

    def finished():
        events["finished"] += 1

    q.on_queue_finished = finished
    yield q, jobs, events, store, tmp_path
    store.close()


def _d(tmp_path, n=0):
    return Download(url=f"https://e.com/{n}", kind=Kind.FILE, filename=f"f{n}.bin",
                    folder=str(tmp_path), added_at=1000.0 + n)


def test_respects_max_active(env):
    q, jobs, _, _, tmp = env
    for i in range(5):
        q.add(_d(tmp, i))
    assert sum(j.started for j in jobs.values()) == 3
    assert len(q.active_ids()) == 3


def test_started_in_order_added(env):
    q, jobs, _, _, tmp = env
    ds = [q.add(_d(tmp, i)) for i in range(5)]
    assert q.active_ids() == {ds[0].id, ds[1].id, ds[2].id}


def test_finish_starts_next(env):
    q, jobs, _, _, tmp = env
    ds = [q.add(_d(tmp, i)) for i in range(4)]
    jobs[ds[0].id].finish(Status.COMPLETED)
    assert ds[3].id in jobs and jobs[ds[3].id].started


def test_start_now_exceeds_limit(env):
    q, jobs, _, _, tmp = env
    ds = [q.add(_d(tmp, i)) for i in range(5)]
    q.start_now(ds[4].id)
    assert jobs[ds[4].id].started
    assert len(q.active_ids()) == 4


def test_pause_then_resume_requeues(env):
    q, jobs, _, store, tmp = env
    d = q.add(_d(tmp))
    q.pause(d.id)
    assert d.status is Status.PAUSED
    assert store.get(d.id).status is Status.PAUSED
    q.resume(d.id)
    assert d.id in q.active_ids()


def test_pause_queued_item(env):
    q, jobs, _, _, tmp = env
    ds = [q.add(_d(tmp, i)) for i in range(4)]
    q.pause(ds[3].id)
    assert ds[3].status is Status.PAUSED
    jobs[ds[0].id].finish(Status.COMPLETED)
    assert ds[3].id not in q.active_ids()


def test_retry_clears_error(env):
    q, jobs, _, store, tmp = env
    d = q.add(_d(tmp))
    jobs[d.id].finish(Status.FAILED, "disk is full")
    q.retry(d.id)
    assert d.error is None
    assert d.id in q.active_ids()


def test_queue_finished_fires_once_when_all_done(env):
    q, jobs, events, _, tmp = env
    a, b = q.add(_d(tmp, 1)), q.add(_d(tmp, 2))
    jobs[a.id].finish(Status.COMPLETED)
    assert events["finished"] == 0
    jobs[b.id].finish(Status.COMPLETED)
    assert events["finished"] == 1


def test_queue_finished_not_fired_when_last_paused(env):
    q, jobs, events, _, tmp = env
    a, b = q.add(_d(tmp, 1)), q.add(_d(tmp, 2))
    jobs[a.id].finish(Status.COMPLETED)
    q.pause(b.id)
    assert events["finished"] == 0


def test_set_max_active_raises_and_pumps(env):
    q, jobs, _, _, tmp = env
    for i in range(5):
        q.add(_d(tmp, i))
    q.set_max_active(5)
    assert len(q.active_ids()) == 5


def test_pause_all_holds_queue_and_resume_all_restarts(env):
    q, jobs, _, _, tmp = env
    ds = [q.add(_d(tmp, i)) for i in range(4)]
    q.pause_all()
    assert q.active_ids() == set()
    assert all(d.status is not Status.DOWNLOADING for d in ds)
    q.resume_all()
    assert len(q.active_ids()) == 3


def test_remove_deletes_part_file(env):
    q, jobs, events, store, tmp = env
    d = q.add(_d(tmp))
    d.part_path.write_bytes(b"partial")
    q.remove(d.id)
    assert not d.part_path.exists()
    assert store.get(d.id) is None
    assert events["removed"] == [d.id]


def test_remove_keeps_finished_file_unless_asked(env):
    q, jobs, _, _, tmp = env
    a, b = q.add(_d(tmp, 1)), q.add(_d(tmp, 2))
    for d in (a, b):
        jobs[d.id].finish(Status.COMPLETED)
        d.target_path.write_bytes(b"done")
    q.remove(a.id)
    q.remove(b.id, delete_file=True)
    assert a.target_path.exists()
    assert not b.target_path.exists()


def test_start_saves_downloading_status(env):
    # Review I2: a crash must find "downloading" in the store, so the item comes back Paused.
    q, jobs, _, store, tmp = env
    d = q.add(_d(tmp))
    assert store.get(d.id).status is Status.DOWNLOADING


def test_pump_starts_waiting_items_loaded_from_store(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    for i in range(4):
        store.add(_d(tmp_path, i))  # QUEUED
    started = []
    q = DownloadQueue(store, lambda d, p, f: type("J", (), {"start": lambda s: started.append(d.id),
                                                          "pause": lambda s: None, "download": d})(),
                      max_active=3)
    q.pump()
    assert len(started) == 3


def test_add_many_starts_up_to_limit(env):
    q, jobs, events, store, tmp = env
    ds = q.add_many([_d(tmp, i) for i in range(10)])
    assert len(store.all()) == 10
    assert len(q.active_ids()) == 3
    assert all(d.id is not None for d in ds)


def test_resume_one_item_while_held_starts_it(env):
    # Review I9: "Pause all", then Resume on one card must start that card.
    q, jobs, _, _, tmp = env
    d = q.add(_d(tmp))
    q.pause_all()
    q.resume(d.id)
    assert d.id in q.active_ids()


def test_add_while_held_starts_it(env):
    q, jobs, _, _, tmp = env
    q.pause_all()
    d = q.add(_d(tmp))
    assert d.id in q.active_ids()


def test_pause_all_signals_every_job_before_waiting(tmp_path):
    # Review I7: stopping N downloads takes as long as the slowest, not the sum.
    store = Store(tmp_path / "db.sqlite")

    class SlowJob:
        def __init__(self, d, p, f):
            self.download, self._finish, self.signalled = d, f, None

        def start(self):
            pass

        def signal_pause(self):
            self.signalled = time.monotonic()

        def pause(self):
            if self.signalled is None:
                self.signal_pause()
            time.sleep(max(0.0, self.signalled + 0.3 - time.monotonic()))
            self.download.status = Status.PAUSED
            self._finish(self.download)

    q = DownloadQueue(store, SlowJob, max_active=3)
    for i in range(3):
        q.add(_d(tmp_path, i))
    t0 = time.monotonic()
    q.pause_all()
    assert time.monotonic() - t0 < 0.6


def test_unique_name_avoids_queue_and_disk(env):
    # Review C1: two links ending in the same name must not share a file.
    q, jobs, _, _, tmp = env
    q.add(_d(tmp, 1))  # f1.bin is now taken by a download
    assert q.unique_name(str(tmp), "f1.bin") == "f1 (2).bin"
    (tmp / "x.zip.part").write_bytes(b"stale")
    assert q.unique_name(str(tmp), "x.zip") == "x (2).zip"
    assert q.unique_name(str(tmp), "free.zip") == "free.zip"


def test_unique_name_by_stem_for_videos(env):
    # Review I4: "Song" as video and as audio must not share a stem.
    q, jobs, _, _, tmp = env
    (tmp / "Song.mp4").write_bytes(b"v")
    assert q.unique_name(str(tmp), "Song.m4a", by_stem=True) == "Song (2).m4a"


def test_remove_youtube_leftovers(env):
    # Review I5: yt-dlp's piece files go too; other files with similar names stay.
    q, jobs, _, _, tmp = env
    d = _d(tmp)
    d.kind, d.filename = Kind.VIDEO, "Clip.mp4"
    q.add(d)
    leftovers = ["Clip.f137.mp4.part", "Clip.f140.m4a", "Clip.f137.mp4.ytdl", "Clip.f137.mp4.part-Frag3",
                 "Clip.temp.mp4"]
    keep = ["Clip (2).mp4", "Clip.mp4", "Clipper.f137.mp4.part",
            "Clip.f1.txt", "Clip.notes.part", "Clip.temp.docx", "Clip.backup.ytdl"]  # the user's own files
    for name in leftovers + keep:
        (tmp / name).write_bytes(b"x")
    q.remove(d.id)
    assert sorted(p.name for p in tmp.iterdir() if p.suffix != ".sqlite" and "sqlite" not in p.name) == sorted(keep)


def test_loads_existing_downloads_from_store(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    d = _d(tmp_path)
    d.status = Status.PAUSED
    store.add(d)
    q = DownloadQueue(store, lambda *a: None)
    assert [x.id for x in q.downloads()] == [d.id]


def test_redownload_deletes_damaged_file_and_requeues(env, tmp_path):
    q, jobs = env[0], env[1]
    from downloader.engine.models import Segment
    d = Download(url="https://x.org/a.zip", kind=Kind.FILE, filename="a.zip", folder=str(tmp_path),
                 status=Status.FAILED, error="checksum doesn't match · the file may be damaged",
                 downloaded=5, segments=[Segment(0, 0, 4, 5)], verified=False, duplicate_of="b.zip")
    q._store.add(d)
    q._downloads[d.id] = d
    (tmp_path / "a.zip.part").write_bytes(b"12345")
    (tmp_path / "a.zip").write_bytes(b"not ours")  # a file that took the name later stays
    q.redownload(d.id)
    assert not (tmp_path / "a.zip.part").exists()
    assert (tmp_path / "a.zip").read_bytes() == b"not ours"
    assert d.segments == [] and d.downloaded == 0 and d.error is None and d.duplicate_of is None
    assert d.status in (Status.QUEUED, Status.DOWNLOADING)
    assert jobs[d.id].started


def test_redownload_ignores_other_statuses(env, tmp_path):
    q = env[0]
    d = Download(url="https://x.org/a.zip", kind=Kind.FILE, filename="a.zip", folder=str(tmp_path),
                 status=Status.COMPLETED)
    q._store.add(d)
    q._downloads[d.id] = d
    (tmp_path / "a.zip").write_bytes(b"12345")
    q.redownload(d.id)
    assert (tmp_path / "a.zip").exists() and d.status is Status.COMPLETED
