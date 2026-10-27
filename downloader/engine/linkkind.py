"""Works out what a pasted link is: a plain file, a YouTube video, or a YouTube playlist."""
from __future__ import annotations

from enum import Enum
from urllib.parse import parse_qs, urlsplit


class LinkKind(Enum):
    FILE = "file"
    VIDEO = "video"
    PLAYLIST = "playlist"
    INVALID = "invalid"


_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}


def classify(text: str) -> LinkKind:
    text = text.strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return LinkKind.INVALID
    if parts.scheme not in ("http", "https") or not parts.hostname or " " in text:
        return LinkKind.INVALID

    host = parts.hostname.lower()
    path = parts.path.rstrip("/")
    if host == "youtu.be":
        return LinkKind.VIDEO if path.strip("/") else LinkKind.INVALID
    if host in _YOUTUBE_HOSTS:
        query = parse_qs(parts.query)
        # A watch link that also names a playlist is still about the one video.
        if path == "/watch" and query.get("v"):
            return LinkKind.VIDEO
        if path.startswith("/shorts/") and len(path) > len("/shorts/"):
            return LinkKind.VIDEO
        if path == "/playlist" and query.get("list"):
            return LinkKind.PLAYLIST
        return LinkKind.INVALID
    return LinkKind.FILE


def video_id(url: str) -> str | None:
    """The YouTube video id, whatever form the link takes; None for anything else."""
    if classify(url) is not LinkKind.VIDEO:
        return None
    parts = urlsplit(url.strip())
    path = parts.path.strip("/")
    if parts.hostname.lower() == "youtu.be":
        return path.split("/")[0] or None
    if path.startswith("shorts/"):
        return path.split("/")[1] or None
    return parse_qs(parts.query)["v"][0]
