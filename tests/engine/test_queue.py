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


def test_loads_existing_downloads_from_store(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    d = _d(tmp_path)
    d.status = Status.PAUSED
    store.add(d)
    q = DownloadQueue(store, lambda *a: None)
    assert [x.id for x in q.downloads()] == [d.id]
