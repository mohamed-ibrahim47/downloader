"""Owns the engine and turns its worker-thread callbacks into Qt signals on the GUI thread."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from ..engine.checksum import Checksum
from ..engine.history import Existing, already_have, find_existing
from ..engine.http_job import HttpJob
from ..engine.linkkind import video_id
from ..engine.models import Download, Kind, Status
from ..engine.paths import safe_filename
from ..engine.probe import FileInfo
from ..engine.queue import DownloadQueue
from ..engine.ratelimit import TokenBucket
from ..engine.settings import Settings
from ..engine.speed import SpeedMeter
from ..engine.store import Store
from ..engine.timecode import clip_suffix
from ..engine.youtube import (MB_PER_MINUTE, FormatChoice, FormatOption, PlaylistInfo, VideoInfo,
                              YouTubeJob, clip_size, selector_for)

CARD_UPDATE_EVERY = 0.1  # at most 10 updates a second per card


class Controller(QObject):
    changed = Signal(int)
    removed = Signal(int)
    finished = Signal(int)  # a download completed or failed
    queue_finished = Signal()
    speed_tick = Signal(float, list)

    _engine_change = Signal(object, object)  # (download, bytes downloaded at that moment)
    _engine_removed = Signal(int)
    _engine_queue_finished = Signal()

    def __init__(self, store: Store, settings: Settings, job_factory=None):
        super().__init__()
        self.store = store
        self.settings = settings
        self.limiter = TokenBucket(settings.speed_limit_bps)
        self.queue = DownloadQueue(store, job_factory or self._make_job, settings.max_active)
        self._meters: dict[int, SpeedMeter] = {}
        self._total = SpeedMeter()
        self._seen_bytes: dict[int, int] = {}
        self._seen_status: dict[int, Status] = {d.id: d.status for d in self.queue.downloads()}
        self._last_emit: dict[int, float] = {}
        self._pending: set[int] = set()

        self._engine_change.connect(self._on_change, Qt.QueuedConnection)
        self._engine_removed.connect(self._on_removed, Qt.QueuedConnection)
        self._engine_queue_finished.connect(self.queue_finished, Qt.QueuedConnection)
        # The byte count travels with the signal: by the time the GUI thread runs, the
        # download object has moved on, and speed is measured from the differences.
        self.queue.on_change = lambda d: self._engine_change.emit(d, d.downloaded)
        self.queue.on_removed = self._engine_removed.emit
        self.queue.on_queue_finished = self._engine_queue_finished.emit

        self._flush_timer = QTimer(self, interval=100, timeout=self._flush)
        self._flush_timer.start()
        self._speed_timer = QTimer(self, interval=500, timeout=self._tick)
        self._speed_timer.start()
        # Downloads left waiting when the app last closed carry on. (Those that were running
        # come back Paused, from the store, and wait for the user.)
        self.queue.pump()

    # ---- jobs ---------------------------------------------------------------

    def _make_job(self, d: Download, on_progress, on_finish):
        if d.kind is Kind.FILE:
            return HttpJob(d, self.store, self.limiter, on_progress, on_finish)
        return YouTubeJob(d, self.limiter, on_progress, on_finish)

    # ---- reading ------------------------------------------------------------

    def downloads(self) -> list[Download]:
        return self.queue.downloads()

    def get(self, download_id: int) -> Download | None:
        return self.queue.get(download_id)

    def speed_of(self, download_id: int) -> tuple[float | None, float | None]:
        d, meter = self.get(download_id), self._meters.get(download_id)
        if d is None or meter is None or d.status is not Status.DOWNLOADING:
            return None, None
        now = time.monotonic()
        remaining = d.size - d.downloaded if d.size else None
        return meter.rate(now), meter.eta(remaining, now)

    # ---- adding -------------------------------------------------------------

    def add_file(self, info: FileInfo, folder: str | None = None, checksum: Checksum | None = None,
                 checksum_source: str | None = None) -> Download:
        folder = folder or self.settings.folder_general
        d = Download(url=info.url, kind=Kind.FILE, filename=self.queue.unique_name(folder, info.filename),
                     folder=folder, size=info.size, resumable=info.resumable,
                     connections=self.settings.connections,
                     checksum=str(checksum) if checksum else None,
                     checksum_source=checksum_source if checksum else None)
        return self.queue.add(d)

    def add_video(self, info: VideoInfo, choice: FormatChoice | FormatOption, folder: str | None = None,
                  subtitles: bool | None = None, start_now: bool = False,
                  clip: tuple[int, int] | None = None) -> Download:
        folder = folder or self.settings.folder_video
        if isinstance(choice, FormatOption):
            quality = "Audio" if choice.id in ("m4a", "mp3") else choice.quality
            ext = {"m4a": "m4a", "mp3": "mp3"}.get(choice.id, "mp4")
            note, audio_format = None, choice.audio_format
        else:
            quality = f"{choice.height}p" if choice.height else "Audio"
            ext, note, audio_format = choice.ext, choice.note, choice.audio_format
        suffix = clip_suffix(*clip) if clip else ""
        name = self.queue.unique_name(folder, f"{safe_filename(info.title)}{suffix}.{ext}", by_stem=True)
        size = clip_size(choice.size, info.duration, *clip) if clip else choice.size
        d = Download(url=info.url, kind=Kind.VIDEO, filename=name, folder=folder, size=size,
                     resumable=True, quality=quality, format_selector=choice.selector, note=note,
                     audio_format=audio_format, title=info.title,
                     subtitles=self.settings.subtitles if subtitles is None else subtitles,
                     video_id=video_id(info.url), clip_start=clip[0] if clip else None,
                     clip_end=clip[1] if clip else None)
        self.queue.add(d)
        if start_now:
            self.queue.start_now(d.id)
        return d

    def add_playlist(self, info: PlaylistInfo, quality: str, picked: list[int]) -> list[Download]:
        folder = str(Path(self.settings.folder_playlist) / safe_filename(info.title))
        ext = "m4a" if quality == "Audio" else "mp4"
        width = 3 if len(info.entries) >= 100 else 2
        batch: list[Download] = []
        names: set[str] = set()
        for e in info.entries:
            if e.index not in picked:
                continue
            title = safe_filename(e.title)
            base = f"{e.index:0{width}d} - {title}" if self.settings.number_playlist_files else title
            name = self.queue.unique_name(folder, f"{base}.{ext}", by_stem=True)
            while Path(name).stem.lower() in names:  # two videos with the same title in this batch
                name = self.queue.unique_name(folder, f"{Path(name).stem} (2).{ext}", by_stem=True)
            names.add(Path(name).stem.lower())
            size = round(e.duration / 60 * MB_PER_MINUTE[quality] * 1_000_000) if e.duration else None
            batch.append(Download(url=e.url, kind=Kind.PLAYLIST_ITEM, filename=name, folder=folder, size=size,
                                  resumable=True, quality=quality, format_selector=selector_for(quality),
                                  title=e.title, playlist_title=info.title, playlist_index=e.index,
                                  subtitles=self.settings.subtitles, video_id=video_id(e.url)))
        return self.queue.add_many(batch)

    # ---- what you already have ----------------------------------------------

    def existing(self, url: str, clip: tuple[int, int] | None = None) -> Existing | None:
        return find_existing(self.downloads(), url, clip)

    def already_have(self, info: PlaylistInfo) -> set[int]:
        return already_have(self.downloads(), info.entries)

    # ---- acting -------------------------------------------------------------

    @staticmethod
    def _background(fn, *args) -> None:
        # Pausing waits for connections to wind down; never make the window wait for that.
        threading.Thread(target=fn, args=args, daemon=True).start()

    def primary(self, download_id: int) -> None:
        d = self.get(download_id)
        if d is None:
            return
        if d.status is Status.DOWNLOADING:
            self._background(self.queue.pause, download_id)
        elif d.status is Status.QUEUED:
            self.queue.start_now(download_id)
        elif d.status is Status.PAUSED:
            self.queue.resume(download_id)
        elif d.status is Status.FAILED:
            self.queue.retry(download_id)
        else:
            self.open_folder(d)

    @staticmethod
    def open_folder(d: Download) -> None:
        if sys.platform == "win32" and d.target_path.exists():
            subprocess.Popen(["explorer", "/select,", str(d.target_path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(d.folder))

    def remove(self, download_id: int, delete_file: bool = False) -> None:
        self._background(self.queue.remove, download_id, delete_file)

    def redownload(self, download_id: int) -> None:
        """Delete a damaged download's file and fetch it again (the window asks first)."""
        self._background(self.queue.redownload, download_id)

    def keep_both(self, download_id: int) -> None:
        d = self.get(download_id)
        if d is None:
            return
        d.duplicate_of = None
        self.store.update(d)
        self.changed.emit(download_id)

    def delete_copy(self, download_id: int) -> None:
        """Remove a download that turned out identical to a file already there, and its file."""
        self._background(self.queue.remove, download_id, True)

    def start_queue(self) -> None:
        self.queue.resume_all()

    def stop_queue(self) -> None:
        self._background(self.queue.pause_all)

    def apply_settings(self, s: Settings) -> None:
        self.settings = s
        self.limiter.set_rate(s.speed_limit_bps)
        self.queue.set_max_active(s.max_active)

    def shutdown(self) -> None:
        """Pause everything so progress is saved, then stop the timers."""
        self._flush_timer.stop()
        self._speed_timer.stop()
        self.queue.pause_all()

    # ---- engine events (GUI thread) ----------------------------------------

    def _on_change(self, d: Download, downloaded: int) -> None:
        now = time.monotonic()
        last = self._seen_bytes.get(d.id)
        if last is not None and downloaded > last:
            self._meters.setdefault(d.id, SpeedMeter()).add(downloaded - last, now)
            self._total.add(downloaded - last, now)
        self._seen_bytes[d.id] = downloaded

        status_changed = self._seen_status.get(d.id) is not d.status
        self._seen_status[d.id] = d.status
        if status_changed or now - self._last_emit.get(d.id, 0) >= CARD_UPDATE_EVERY:
            self._emit(d.id, now)
        else:
            self._pending.add(d.id)
        if status_changed and d.status in (Status.COMPLETED, Status.FAILED):
            self.finished.emit(d.id)

    def _emit(self, download_id: int, now: float) -> None:
        self._last_emit[download_id] = now
        self._pending.discard(download_id)
        self.changed.emit(download_id)

    def _flush(self) -> None:
        now = time.monotonic()
        for download_id in list(self._pending):
            self._emit(download_id, now)

    def _on_removed(self, download_id: int) -> None:
        for table in (self._meters, self._seen_bytes, self._seen_status, self._last_emit):
            table.pop(download_id, None)
        self._pending.discard(download_id)
        self.removed.emit(download_id)

    def _tick(self) -> None:
        now = time.monotonic()
        self.speed_tick.emit(self._total.rate(now), self._total.history(now))
