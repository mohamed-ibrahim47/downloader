"""Download failures, worded for people rather than programmers."""
from __future__ import annotations

CONNECTION_LOST = "connection lost · check your internet"
DISK_FULL = "disk is full"
CHECKSUM_MISMATCH = "checksum doesn't match · the file may be damaged"
CANT_READ_TO_CHECK = "couldn't read the downloaded file to check it"


class DownloadError(Exception):
    """A failure whose message is shown on the download card as-is."""


def message_for_status(code: int) -> str:
    if code == 403:
        return "server refused (403) · link may have expired"
    if code == 404:
        return "file not found on server (404)"
    if code == 429:
        return "server is busy (429) · try again later"
    return f"server error ({code})"


def cant_write(folder) -> str:
    return f"can't write to {folder}"
