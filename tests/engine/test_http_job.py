import os
import threading
import time

import pytest

from downloader.engine.http_job import HttpJob
from downloader.engine.models import Download, Kind, Status
from downloader.engine.probe import probe
from downloader.engine.ratelimit import TokenBucket
from downloader.engine.store import Store

DATA = os.urandom(5_000_000)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    yield s
    s.close()


def _download(url, folder, connections=8, use_probe=True, filename="file.bin"):
    if use_probe:
        info = probe(url)
        d = Download(url=url, kind=Kind.FILE, filename=info.filename, folder=str(folder),
                     size=info.size, resumable=info.resumable, connections=connections)
    else:
        d = Download(url=url, kind=Kind.FILE, filename=filename, folder=str(folder), connections=connections)
    return d


def _job(d, store, finished=None):
    if d.id is None:
        store.add(d)
    return HttpJob(d, store, TokenBucket(None), on_progress=lambda _d: None,
                   on_finish=(finished.append if finished is not None else (lambda _d: None)),
                   backoff=(0, 0, 0))


def test_segmented_download_matches_bytes(http_server, store, tmp_path):
    url = http_server.add("/big.bin", DATA)
    d = _download(url, tmp_path / "out")
    finished = []
    _job(d, store, finished).run()
    assert d.status is Status.COMPLETED
    assert (tmp_path / "out" / "big.bin").read_bytes() == DATA
    assert not (tmp_path / "out" / "big.bin.part").exists()
    assert len({r for r in http_server.get_ranges("/big.bin")}) >= 2
    assert finished == [d]
    assert store.get(d.id).status is Status.COMPLETED


def test_no_range_falls_back_to_single_connection(http_server, store, tmp_path):
    url = http_server.add("/plain.bin", DATA, ranges=False)
    d = _download(url, tmp_path)
    before = len(http_server.get_ranges("/plain.bin"))
    _job(d, store).run()
    assert d.status is Status.COMPLETED
    assert d.resumable is False
    assert (tmp_path / "plain.bin").read_bytes() == DATA
    assert http_server.get_ranges("/plain.bin")[before:] == [None]


def test_server_ignoring_range_mid_download_falls_back(http_server, store, tmp_path):
    url = http_server.add("/liar.bin", DATA, ranges=False)
    d = _download(url, tmp_path)
    d.resumable = True  # the server claimed ranges but answers 200 to them
    _job(d, store).run()
    assert d.status is Status.COMPLETED
    assert d.resumable is False
    assert (tmp_path / "liar.bin").read_bytes() == DATA


def test_unknown_length_completes(http_server, store, tmp_path):
    url = http_server.add("/nolen.bin", DATA, ranges=False, length=False)
    d = _download(url, tmp_path)
    assert d.size is None
    _job(d, store).run()
    assert d.status is Status.COMPLETED
    assert (tmp_path / "nolen.bin").read_bytes() == DATA
    assert d.size == len(DATA)


def test_pause_then_resume_in_new_instance(http_server, store, tmp_path):
    url = http_server.add("/slow.bin", DATA, delay_per_chunk=0.03)
    d = _download(url, tmp_path, connections=2)
    job = _job(d, store)
    job.start()
    deadline = time.monotonic() + 10
    while d.downloaded == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    job.pause()
    assert d.status is Status.PAUSED
    saved = store.get(d.id)
    assert saved.status is Status.PAUSED
    assert sum(s.done for s in saved.segments) > 0
    assert saved.downloaded < len(DATA)

    http_server.entries["/slow.bin"].delay_per_chunk = 0
    before = len(http_server.get_ranges("/slow.bin"))
    _job(saved, store).run()
    assert saved.status is Status.COMPLETED
    assert (tmp_path / "slow.bin").read_bytes() == DATA
    resumed = http_server.get_ranges("/slow.bin")[before:]
    assert resumed and all(not r.startswith("bytes=0-") for r in resumed)


def test_retries_then_succeeds(http_server, store, tmp_path):
    url = http_server.add("/flaky.bin", DATA, fail_times=2)
    d = _download(url, tmp_path, connections=1)
    _job(d, store).run()
    assert d.status is Status.COMPLETED
    assert (tmp_path / "flaky.bin").read_bytes() == DATA


def test_gives_up_after_three_retries(http_server, store, tmp_path):
    url = http_server.add("/broken.bin", DATA, fail_times=99)
    d = _download(url, tmp_path, connections=1)
    before = len(http_server.get_ranges("/broken.bin"))
    _job(d, store).run()
    assert d.status is Status.FAILED
    assert d.error == "connection lost · check your internet"
    assert len(http_server.get_ranges("/broken.bin")) - before == 4


def test_403_fails_with_plain_message(http_server, store, tmp_path):
    url = http_server.add("/gone.bin", DATA, status=403)
    d = _download(url, tmp_path, use_probe=False)
    finished = []
    _job(d, store, finished).run()
    assert d.status is Status.FAILED
    assert d.error == "server refused (403) · link may have expired"
    assert len(http_server.get_ranges("/gone.bin")) == 1
    assert finished == [d]


def test_existing_target_gets_suffix(http_server, store, tmp_path):
    (tmp_path / "dup.bin").write_bytes(b"old")
    url = http_server.add("/dup.bin", DATA)
    d = _download(url, tmp_path)
    _job(d, store).run()
    assert d.filename == "dup (2).bin"
    assert (tmp_path / "dup.bin").read_bytes() == b"old"
    assert (tmp_path / "dup (2).bin").read_bytes() == DATA


def test_missing_folder_created(http_server, store, tmp_path):
    url = http_server.add("/m.bin", DATA)
    d = _download(url, tmp_path / "a" / "b")
    _job(d, store).run()
    assert (tmp_path / "a" / "b" / "m.bin").read_bytes() == DATA


def test_unwritable_folder_fails_plainly(http_server, store, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"i am a file")
    url = http_server.add("/w.bin", DATA)
    d = _download(url, blocker)
    _job(d, store).run()
    assert d.status is Status.FAILED
    assert d.error.startswith("can't write to")


def test_speed_limit_slows_download(http_server, store, tmp_path):
    data = os.urandom(600_000)
    url = http_server.add("/lim.bin", data)
    d = _download(url, tmp_path, connections=1)
    store.add(d)
    job = HttpJob(d, store, TokenBucket(400_000), lambda _d: None, lambda _d: None, backoff=(0, 0, 0))
    t0 = time.monotonic()
    job.run()
    assert d.status is Status.COMPLETED
    assert time.monotonic() - t0 >= 0.4
