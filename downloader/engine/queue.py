"""Decides which downloads run: a few at a time, oldest first, with "start now" to jump the line."""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Callable, Protocol

from .models import Download, Kind, Status
from .store import Store


class Job(Protocol):
    download: Download

    def start(self) -> None: ...
    def pause(self) -> None: ...


JobFactory = Callable[[Download, Callable[[Download], None], Callable[[Download], None]], Job]

MEDIA_EXTS = (".mp4", ".m4a", ".mp3", ".webm", ".mkv", ".opus")
# Exactly what yt-dlp leaves next to `<stem>.<ext>` while working, always ending in a media type:
# format pieces (f137.mp4 / .part / .part-Frag3 / .ytdl), the merge's temp.mp4, mp4.part, mp4.ytdl.
_MEDIA = r"(mp4|m4a|webm|mkv|opus|mp3)"
_LEFTOVER = re.compile(
    rf"^(f\d+(-\d+)?\.{_MEDIA}(\.part(-Frag\d+)?|\.ytdl)?|temp\.{_MEDIA}|{_MEDIA}\.part(-Frag\d+)?|{_MEDIA}\.ytdl)$")


def _noop(*_args) -> None:
    pass


def leftover_files(d: Download) -> list[Path]:
    """Unfinished pieces of a download: its .part, and for YouTube yt-dlp's working files."""
    files = [d.part_path] if d.part_path.exists() else []
    if d.kind is not Kind.FILE:
        stem = Path(d.filename).stem
        folder = Path(d.folder)
        if folder.is_dir():
            prefix = stem + "."
            for p in folder.iterdir():
                if p.name.startswith(prefix) and _LEFTOVER.match(p.name[len(prefix):]):
                    files.append(p)
    return files


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

    def unique_name(self, folder: str, name: str, by_stem: bool = False) -> str:
        """`name`, or `name (2)`… so that no file on disk, no .part, and no other download in the
        queue uses it. With `by_stem`, any media file sharing the stem counts as taken (yt-dlp
        names its output by stem and picks the extension itself)."""
        folder_path = Path(folder)
        stem, ext = Path(name).stem, Path(name).suffix
        with self._lock:
            taken = {d.filename.lower() for d in self._downloads.values()
                     if Path(d.folder) == folder_path}
        taken_stems = {Path(n).stem for n in taken}

        def free(candidate_stem: str) -> bool:
            candidate = f"{candidate_stem}{ext}"
            if candidate.lower() in taken or (folder_path / candidate).exists() \
                    or (folder_path / f"{candidate}.part").exists():
                return False
            if by_stem:
                if candidate_stem.lower() in taken_stems:
                    return False
                if any((folder_path / f"{candidate_stem}{e}").exists() for e in MEDIA_EXTS):
                    return False
            return True

        if free(stem):
            return name
        n = 2
        while not free(f"{stem} ({n})"):
            n += 1
        return f"{stem} ({n}){ext}"

    # ---- changing -----------------------------------------------------------

    def add(self, d: Download) -> Download:
        return self.add_many([d])[0]

    def add_many(self, downloads: list[Download]) -> list[Download]:
        """Queue downloads with one database write. Adding is the user asking for these now,
        so it lifts a "Pause all"."""
        for d in downloads:
            d.status = Status.QUEUED
        with self._lock:
            self._store.add_many(downloads)
            for d in downloads:
                self._downloads[d.id] = d
            self._held = False
        for d in downloads:
            self.on_change(d)
        self.pump()
        return downloads

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
        try:
            for path in leftover_files(d):
                path.unlink(missing_ok=True)
            if delete_file and d.status is Status.COMPLETED:
                d.target_path.unlink(missing_ok=True)
        except OSError:
            pass  # a file still open elsewhere stays behind; the download itself is gone
        self.on_removed(download_id)

    def pause_all(self) -> None:
        with self._lock:
            self._held = True
            jobs = list(self._jobs.values())
        # Ask every job to stop first, then wait: the wait is the slowest job's, not the sum.
        for job in jobs:
            signal = getattr(job, "signal_pause", None)
            if signal:
                signal()
        for job in jobs:
            job.pause()

    def resume_all(self) -> None:
        with self._lock:
            self._held = False
            for d in self._downloads.values():
                if d.status is Status.PAUSED:
                    self._set(d, Status.QUEUED)
        self.pump()

    def set_max_active(self, n: int) -> None:
        self._max_active = max(1, n)
        self.pump()

    def pump(self) -> None:
        """Start waiting downloads while there are free slots (unless "Pause all" holds the queue)."""
        with self._lock:
            if self._held:
                return
            waiting = sorted((d for d in self._downloads.values() if d.status is Status.QUEUED),
                             key=lambda d: (d.added_at, d.id))
            for d in waiting:
                if len(self._jobs) >= self._max_active:
                    break
                self._start(d)

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
            if self._held and len(self._jobs) < self._max_active:
                # The user asked for this one: start it even while the rest stay paused.
                self._start(d)
                return
        self.pump()

    def _start(self, d: Download) -> None:
        job = self._factory(d, self._progress, self._finished)
        self._jobs[d.id] = job
        # Saved, so a crash finds it "downloading" and brings it back Paused.
        self._set(d, Status.DOWNLOADING)
        job.start()

    def _progress(self, d: Download) -> None:
        self.on_change(d)

    def _finished(self, d: Download) -> None:
        with self._lock:
            self._jobs.pop(d.id, None)
            if d.id in self._downloads:
                self._store.update(d)
        self.on_change(d)
        self.pump()
        with self._lock:
            idle = not self._jobs and not any(x.status is Status.QUEUED for x in self._downloads.values())
        if idle and d.status is not Status.PAUSED:
            self.on_queue_finished()
