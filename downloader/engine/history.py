"""Have you got this already? Matches a link against what's in the downloads list."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .linkkind import video_id
from .models import Download, Kind, Status
from .youtube import PlaylistEntry


@dataclass
class Existing:
    download: Download
    finished: bool  # completed and the file is still there; otherwise still in the list


def same_link(a: str, b: str) -> bool:
    """Same URL apart from the #fragment (query strings often matter for files)."""
    return a.split("#", 1)[0] == b.split("#", 1)[0]


def _video_of(d: Download) -> str | None:
    if d.kind is Kind.FILE:
        return None
    return d.video_id or video_id(d.url)


def find_existing(downloads: Iterable[Download], url: str,
                  clip: tuple[int, int] | None = None) -> Existing | None:
    """The download you already have for this link. For a video, any download of it counts (a
    clip, or the whole video), the same part first; a finished one beats one still in the list."""
    vid = video_id(url)
    finished, in_list = [], []
    for d in downloads:
        if vid:
            if _video_of(d) != vid:
                continue
        elif d.kind is not Kind.FILE or not same_link(d.url, url):
            continue
        if d.status is Status.COMPLETED:
            if d.target_path.exists():
                finished.append(Existing(d, True))
        else:
            in_list.append(Existing(d, False))
    for group in (finished, in_list):
        if group:
            return next((e for e in group if e.download.clip == clip), group[0])
    return None


def already_have(downloads: Iterable[Download], entries: Iterable[PlaylistEntry]) -> set[int]:
    """Playlist entries whose whole video is already downloaded and still on disk."""
    have = {_video_of(d) for d in downloads
            if d.status is Status.COMPLETED and d.clip is None and d.target_path.exists()}
    have.discard(None)
    return {e.index for e in entries if video_id(e.url) in have}
