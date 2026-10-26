"""The download record the whole app passes around."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit


class Status(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"


class Kind(str, Enum):
    FILE = "file"
    VIDEO = "video"
    PLAYLIST_ITEM = "playlist_item"


@dataclass
class Segment:
    """One byte range of a file. `end` is inclusive; -1 means the length is unknown."""

    index: int
    start: int
    end: int
    done: int = 0

    @property
    def finished(self) -> bool:
        return self.end >= 0 and self.start + self.done > self.end


@dataclass
class Download:
    url: str
    kind: Kind
    filename: str
    folder: str
    id: int | None = None
    size: int | None = None
    status: Status = Status.QUEUED
    resumable: bool = False
    connections: int = 8
    added_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    note: str | None = None
    downloaded: int = 0
    quality: str | None = None
    format_selector: str | None = None
    audio_format: str | None = None
    subtitles: bool = False
    title: str | None = None
    playlist_title: str | None = None
    playlist_index: int | None = None
    segments: list[Segment] = field(default_factory=list)

    @property
    def target_path(self) -> Path:
        return Path(self.folder) / self.filename

    @property
    def part_path(self) -> Path:
        return Path(self.folder) / (self.filename + ".part")

    @property
    def host(self) -> str:
        return urlsplit(self.url).hostname or ""
