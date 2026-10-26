"""Asks a server about a file before downloading it: name, size, and whether it can resume."""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import requests

from .errors import CONNECTION_LOST, DownloadError, message_for_status
from .paths import safe_filename

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) downloader/1.0"


@dataclass
class FileInfo:
    url: str
    filename: str
    size: int | None
    resumable: bool


def filename_from_disposition(header: str | None) -> str | None:
    if not header:
        return None
    m = re.search(r"filename\*\s*=\s*([^']*)'[^']*'([^;]+)", header, re.I)
    if m:
        charset = (m.group(1) or "utf-8").strip().lower()
        if charset not in ("utf-8", "iso-8859-1"):
            charset = "utf-8"
        return unquote(m.group(2).strip().strip('"'), encoding=charset, errors="replace")
    m = re.search(r'filename\s*=\s*"([^"]*)"', header, re.I) or re.search(r"filename\s*=\s*([^;]+)", header, re.I)
    return m.group(1).strip() if m else None


def filename_from_url(url: str) -> str:
    parts = urlsplit(url)
    last = unquote(parts.path.rstrip("/").rsplit("/", 1)[-1])
    return last or (parts.hostname or "download")


def _size_from_headers(r: requests.Response) -> int | None:
    content_range = r.headers.get("Content-Range", "")
    m = re.search(r"/(\d+)\s*$", content_range)
    if m:
        return int(m.group(1))
    if r.status_code == 200 and r.headers.get("Content-Length", "").isdigit():
        if r.headers.get("Content-Encoding", "identity") in ("identity", ""):
            return int(r.headers["Content-Length"])
    return None


def probe(url: str, timeout: float = 15) -> FileInfo:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    try:
        head = session.head(url, allow_redirects=True, timeout=timeout)
        size = resumable = None
        final_url, disposition = url, None
        if head.status_code < 400:
            final_url = head.url
            disposition = head.headers.get("Content-Disposition")
            size = _size_from_headers(head)
            resumable = head.headers.get("Accept-Ranges", "").lower() == "bytes"
        if head.status_code >= 400 or not resumable:
            # Some servers refuse HEAD or don't advertise ranges: ask for the first byte instead.
            with session.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                             allow_redirects=True, timeout=timeout) as r:
                if r.status_code >= 400:
                    raise DownloadError(message_for_status(r.status_code))
                final_url = r.url
                disposition = r.headers.get("Content-Disposition") or disposition
                resumable = r.status_code == 206
                size = _size_from_headers(r) if (resumable or size is None) else size
    except requests.RequestException as e:
        raise DownloadError(CONNECTION_LOST) from e
    finally:
        session.close()

    name = filename_from_disposition(disposition) or filename_from_url(final_url)
    return FileInfo(url=final_url, filename=safe_filename(name), size=size, resumable=bool(resumable))
