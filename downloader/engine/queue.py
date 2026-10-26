"""Decides which downloads run: a few at a time, oldest first, with "start now" to jump the line."""
from __future__ import annotations

import threading
from typing import Callable, Protocol

from .models import Download, Status
from .store import Store


class Job(Protocol):
    download: Download

    def start(self) -> None: ...
    def pause(self) -> None: ...


JobFactory = Callable[[Download, Callable[[Download], None], Callable[[Download], None]], Job]


def _noop(*_args) -> None:
    pass


class DownloadQueue:
    def __init__(self, store: Store, job_factory: JobFactory, max_active: int = 3):
        self._store = store
        self._factory = job_factory
        self._max_active = max_active
        self._lock = threading.RLock()
        self._jobs: dict[int, Job] = {}
        self._downloads: dict[int, Download] = {d.id: d for d in store.all()}
        self._held = False
        self.on_change: Callable[[Download], None] = _noop
        self.on_removed: Callable[[int], None] = _noop
        self.on_queue_finished: Callable[[], None] = _noop

    # ---- reading ------------------------------------------------------------

    def downloads(self) -> list[Download]:
        with self._lock:
            return sorted(self._downloads.values(), key=lambda d: (d.added_at, d.id), reverse=True)

    def get(self, download_id: int) -> Download | None:
        return self._downloads.get(download_id)

    def active_ids(self) -> set[int]:
        with self._lock:
            return set(self._jobs)

    # ---- changing -----------------------------------------------------------

    def add(self, d: Download) -> Download:
        d.status = Status.QUEUED
        with self._lock:
            self._store.add(d)
            self._downloads[d.id] = d
        self.on_change(d)
        self._pump()
        return d

    def start_now(self, download_id: int) -> None:
        with self._lock:
            d = self._downloads.get(download_id)
            if d is None or download_id in self._jobs or d.status is Status.COMPLETED:
                return
            self._start(d)

    def pause(self, download_id: int) -> None:
        with self._lock:
            job = self._jobs.get(download_id)
            d = self._downloads.get(download_id)
            if job is None and d is not None and d.status is Status.QUEUED:
                self._set(d, Status.PAUSED)
        if job is not None:
            # Outside the lock: the job reports back through _finished from its own thread.
            job.pause()

    def resume(self, download_id: int) -> None:
        self._requeue(download_id, (Status.PAUSED,))

    def retry(self, download_id: int) -> None:
        self._requeue(download_id, (Status.FAILED,))

    def remove(self, download_id: int, delete_file: bool = False) -> None:
        self.pause(download_id)
        with self._lock:
            d = self._downloads.pop(download_id, None)
            if d is None:
                return
            self._store.delete(download_id)
        d.part_path.unlink(missing_ok=True)
        if delete_file and d.status is Status.COMPLETED:
            d.target_path.unlink(missing_ok=True)
        self.on_removed(download_id)

    def pause_all(self) -> None:
        with self._lock:
            self._held = True
            jobs = list(self._jobs.values())
        for job in jobs:
            job.pause()

    def resume_all(self) -> None:
        with self._lock:
            self._held = False
            for d in self._downloads.values():
                if d.status is Status.PAUSED:
                    self._set(d, Status.QUEUED)
        self._pump()

    def set_max_active(self, n: int) -> None:
        self._max_active = max(1, n)
        self._pump()

    # ---- internals ----------------------------------------------------------

    def _set(self, d: Download, status: Status) -> None:
        d.status = status
        self._store.update(d)
        self.on_change(d)

    def _requeue(self, download_id: int, from_statuses: tuple[Status, ...]) -> None:
        with self._lock:
            d = self._downloads.get(download_id)
            if d is None or d.status not in from_statuses:
                return
            d.error = None
            self._set(d, Status.QUEUED)
        self._pump()

    def _start(self, d: Download) -> None:
        job = self._factory(d, self._progress, self._finished)
        self._jobs[d.id] = job
        d.status = Status.DOWNLOADING
        self.on_change(d)
        job.start()

    def _pump(self) -> None:
        with self._lock:
            if self._held:
                return
            waiting = sorted((d for d in self._downloads.values() if d.status is Status.QUEUED),
                             key=lambda d: (d.added_at, d.id))
            for d in waiting:
                if len(self._jobs) >= self._max_active:
                    break
                self._start(d)

    def _progress(self, d: Download) -> None:
        self.on_change(d)

    def _finished(self, d: Download) -> None:
        with self._lock:
            self._jobs.pop(d.id, None)
            if d.id in self._downloads:
                self._store.update(d)
        self.on_change(d)
        self._pump()
        with self._lock:
            idle = not self._jobs and not any(x.status is Status.QUEUED for x in self._downloads.values())
        if idle and d.status is not Status.PAUSED:
            self.on_queue_finished()
