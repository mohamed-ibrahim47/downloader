"""Downloads one file over several connections at once, each fetching its own byte range.

Progress is kept per segment in the store, so a paused or interrupted download picks up
where each connection stopped, even after the app restarts.
"""
from __future__ import annotations

import errno
import threading
import time
from typing import Callable

import requests

from .errors import CONNECTION_LOST, DISK_FULL, DownloadError, cant_write, message_for_status
from .models import Download, Segment, Status
from .paths import unique_path
from .probe import USER_AGENT
from .ratelimit import TokenBucket
from .segments import plan_segments
from .store import Store

__all__ = ["HttpJob", "DownloadError"]

CHUNK = 65536
SAVE_EVERY = 1.0
PROGRESS_EVERY = 0.1


class _NoRanges(Exception):
    """The server answered a range request with the whole file."""


class _HttpStatus(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


class HttpJob:
    def __init__(self, download: Download, store: Store, limiter: TokenBucket,
                 on_progress: Callable[[Download], None], on_finish: Callable[[Download], None],
                 backoff: tuple[float, ...] = (1, 2, 4)):
        self.download = download
        self._store = store
        self._limiter = limiter
        self._on_progress = on_progress
        self._on_finish = on_finish
        self._backoff = backoff
        self._lock = threading.Lock()
        self._pause = threading.Event()
        self._halt = threading.Event()
        self._error: str | None = None
        self._no_ranges = False
        self._thread: threading.Thread | None = None
        self._last_progress = 0.0
        # How many connections may be open at once; drops to 1 if the server says it's busy.
        self._gate = threading.Condition()
        self._allowed = download.connections
        self._open = 0

    # ---- public -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name=f"download-{self.download.id}", daemon=True)
        self._thread.start()

    def signal_pause(self) -> None:
        """Ask to stop without waiting (pausing many downloads: signal all, then wait)."""
        self._pause.set()

    def pause(self) -> None:
        self._pause.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join()

    def run(self) -> None:
        d = self.download
        d.status, d.error = Status.DOWNLOADING, None
        self._store.update(d)
        try:
            self._prepare()
            while True:
                try:
                    self._run_segments()
                    break
                except _NoRanges:
                    d.resumable = False
                    d.segments = plan_segments(d.size, 1, False)
                    self._reset_part()
            if self._pause.is_set():
                d.status = Status.PAUSED
            else:
                self._complete()
        except DownloadError as e:
            d.status, d.error = Status.FAILED, str(e)
        d.downloaded = sum(s.done for s in d.segments)
        self._store.update(d)
        self._on_finish(d)

    # ---- steps --------------------------------------------------------------

    def _prepare(self) -> None:
        d = self.download
        folder = d.target_path.parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if not d.segments:
                d.segments = plan_segments(d.size, d.connections, d.resumable)
            fresh = not any(s.done for s in d.segments)
            if not d.resumable or fresh or not d.part_path.exists():
                # Nothing saved yet (a stale .part of the same name is not ours), or no range
                # support: start over from an empty file.
                for s in d.segments:
                    s.done = 0
                self._reset_part()
        except OSError as e:
            raise DownloadError(DISK_FULL if e.errno == errno.ENOSPC else cant_write(folder)) from e

    def _reset_part(self) -> None:
        d = self.download
        with open(d.part_path, "wb") as f:
            if d.size and d.resumable:
                f.truncate(d.size)

    def _run_segments(self) -> None:
        self._halt.clear()
        self._error, self._no_ranges = None, False
        pending = [s for s in self.download.segments if not s.finished]
        threads = [threading.Thread(target=self._worker, args=(s,), daemon=True) for s in pending]
        for t in threads:
            t.start()
        last_save = time.monotonic()
        while any(t.is_alive() for t in threads):
            threads[0].join(0.2) if threads[0].is_alive() else time.sleep(0.05)
            if time.monotonic() - last_save >= SAVE_EVERY:
                self._save_progress()
                last_save = time.monotonic()
        self._save_progress()
        if self._no_ranges:
            raise _NoRanges
        if self._error:
            raise DownloadError(self._error)

    def _complete(self) -> None:
        d = self.download
        if not all(s.finished for s in d.segments):
            raise DownloadError("download stopped before the end · press Retry to continue")
        if d.size is None:
            d.size = sum(s.done for s in d.segments)
        final = unique_path(d.target_path.parent, d.filename)
        try:
            d.part_path.replace(final)
        except OSError as e:
            raise DownloadError(cant_write(final.parent)) from e
        d.filename = final.name
        d.status = Status.COMPLETED
        d.finished_at = time.time()

    def _save_progress(self) -> None:
        d = self.download
        with self._lock:
            d.downloaded = sum(s.done for s in d.segments)
            segments = [Segment(s.index, s.start, s.end, s.done) for s in d.segments]
        self._store.save_segments(d.id, segments)

    # ---- one connection -----------------------------------------------------

    def _stopped(self) -> bool:
        return self._pause.is_set() or self._halt.is_set()

    def _fail(self, message: str) -> None:
        with self._lock:
            self._error = self._error or message
        self._halt.set()

    def _acquire(self) -> bool:
        with self._gate:
            while self._open >= self._allowed:
                if self._stopped():
                    return False
                self._gate.wait(0.1)
            self._open += 1
            return True

    def _release(self) -> None:
        with self._gate:
            self._open -= 1
            self._gate.notify_all()

    def _fewer_connections(self) -> bool:
        """The server refused extra connections: carry on one at a time. False if already there."""
        with self._gate:
            if self._allowed == 1:
                return False
            self._allowed = 1
            return True

    def _worker(self, seg: Segment) -> None:
        attempt = 0
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        try:
            while not self._stopped():
                if not self._acquire():
                    return
                before = seg.done
                try:
                    self._fetch(session, seg)
                    return
                except _NoRanges:
                    self._no_ranges = True
                    self._halt.set()
                    return
                except _HttpStatus as e:
                    if e.code in (429, 503) and self._fewer_connections():
                        continue
                    retryable = e.code >= 500 or e.code == 429
                    if not retryable or attempt >= len(self._backoff):
                        self._fail(message_for_status(e.code))
                        return
                except (requests.RequestException, ConnectionError):
                    if attempt >= len(self._backoff):
                        self._fail(CONNECTION_LOST)
                        return
                except OSError as e:
                    self._fail(DISK_FULL if e.errno == errno.ENOSPC else cant_write(self.download.folder))
                    return
                except Exception as e:  # never let a connection die silently
                    self._fail(f"unexpected problem: {e}")
                    return
                finally:
                    self._release()
                if seg.done > before:
                    attempt = 0  # it was making progress: a new drop gets fresh retries
                self._pause.wait(self._backoff[attempt])  # pausing cuts the wait short
                attempt += 1
        finally:
            session.close()

    def _fetch(self, session: requests.Session, seg: Segment) -> None:
        d = self.download
        pos = seg.start + seg.done
        headers = {}
        if d.resumable:
            headers["Range"] = f"bytes={pos}-{seg.end if seg.end >= 0 else ''}"
        with session.get(d.url, headers=headers, stream=True, timeout=(15, 30)) as r:
            if r.status_code >= 400:
                raise _HttpStatus(r.status_code)
            if "Range" in headers and r.status_code == 200 and (pos > 0 or len(d.segments) > 1):
                raise _NoRanges
            with open(d.part_path, "r+b") as f:
                f.seek(pos)
                for chunk in r.iter_content(CHUNK):
                    if self._stopped():
                        return
                    if seg.end >= 0:
                        chunk = chunk[: seg.end + 1 - (seg.start + seg.done)]
                    self._limiter.consume(len(chunk))
                    f.write(chunk)
                    with self._lock:
                        seg.done += len(chunk)
                    self._report()
                    if seg.finished:
                        return
        if seg.end < 0:
            # Unknown length: the server closing the connection marks the end.
            seg.end = seg.start + seg.done - 1
            return
        if not seg.finished:
            raise requests.ConnectionError("connection closed early")

    def _report(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_progress < PROGRESS_EVERY:
                return
            self._last_progress = now
            self.download.downloaded = sum(s.done for s in self.download.segments)
        self._on_progress(self.download)
