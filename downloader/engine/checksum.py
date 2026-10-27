"""Checksums: read one pasted by the user or published next to a file, and hash a file to compare."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote, urlsplit, urlunsplit

import requests
import urllib3

from .probe import USER_AGENT

ALGOS = {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}
LABELS = {"md5": "MD5", "sha1": "SHA-1", "sha256": "SHA-256", "sha512": "SHA-512"}
SUMS_FILES = ("{name}.sha256", "{name}.sha256sum", "SHA256SUMS", "sha256sums.txt", "SHA512SUMS", "{name}.sha512")
READ_SIZE = 1 << 20

_PASTED = re.compile(r"^(?:(md5|sha1|sha256|sha512):)?([0-9a-f]+)(?:\s+\*?\S.*)?$", re.I)
_LINE = re.compile(r"^([0-9a-f]+)(?:\s+\*?(.+))?$", re.I)


@dataclass(frozen=True)
class Checksum:
    algo: str
    hex: str

    def __str__(self) -> str:
        return f"{self.algo}:{self.hex}"

    @classmethod
    def from_stored(cls, s: str | None) -> Checksum | None:
        return parse_checksum(s) if s else None

    @property
    def label(self) -> str:
        return LABELS[self.algo]


def _of(hex_text: str) -> Checksum | None:
    algo = ALGOS.get(len(hex_text))
    return Checksum(algo, hex_text.lower()) if algo else None


def parse_checksum(text: str) -> Checksum | None:
    """A bare hash, `sha256:<hash>`, or a line copied from a sums file; the kind comes from its length."""
    m = _PASTED.match(text.strip())
    if not m:
        return None
    found = _of(m.group(2))
    if found is None or (m.group(1) and m.group(1).lower() != found.algo):
        return None
    return found


def find_in_sums(text: str, filename: str, single_file: bool) -> Checksum | None:
    """The checksum a sums file gives for `filename`. A file made for one download
    (`app.zip.sha256`) may hold just the bare hash."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name = m.group(2)
        if name is None:
            if single_file:
                return _of(m.group(1))
            continue
        name = name.strip().lstrip("*")
        if name.startswith("./"):
            name = name[2:]
        if name == filename:
            return _of(m.group(1))
    return None


def hash_file(path: Path, algos: Iterable[str],
              should_stop: Callable[[], bool] = lambda: False) -> dict[str, str] | None:
    """Every requested hash in one read of the file; None if asked to stop part-way."""
    hashers = {a: hashlib.new(a) for a in algos}
    with open(path, "rb") as f:
        while True:
            if should_stop():
                return None
            block = f.read(READ_SIZE)
            if not block:
                break
            for h in hashers.values():
                h.update(block)
    return {a: h.hexdigest() for a, h in hashers.items()}


class _Unreachable(Exception):
    """The site can't be reached at all: no point asking for the other names."""


def _fetch(url: str, timeout: float, limit: int) -> str | None:
    """The text at `url`, or None: no redirects (they may lead off the site), small, and quick."""
    deadline = time.monotonic() + timeout
    try:
        with requests.get(url, allow_redirects=False, stream=True, timeout=timeout,
                          headers={"User-Agent": USER_AGENT}) as r:
            if r.status_code != 200:
                return None
            data = bytearray()
            # read1 returns whatever has arrived, so a server trickling a byte at a time
            # (each just inside the read timeout) still meets the deadline.
            while block := r.raw.read1(65536, decode_content=True):
                data += block
                if len(data) > limit or time.monotonic() > deadline:
                    return None
            return data.decode("utf-8", errors="replace")
    except requests.ConnectionError as e:  # includes ConnectTimeout; a slow read is a ReadTimeout
        raise _Unreachable from e
    except (requests.RequestException, urllib3.exceptions.HTTPError, OSError):
        return None  # a slow or broken reply (raw reads raise urllib3's own errors)


def lookup_sums(url: str, filename: str, timeout: float = 10.0,
                limit: int = 1_000_000) -> tuple[Checksum, str] | None:
    """Look next to the download (same site, same folder) for a published checksum."""
    parts = urlsplit(url)
    folder = parts.path.rsplit("/", 1)[0]
    for pattern in SUMS_FILES:
        sums_name = pattern.format(name=filename)
        sums_url = urlunsplit((parts.scheme, parts.netloc, f"{folder}/{quote(sums_name)}", "", ""))
        try:
            text = _fetch(sums_url, timeout, limit)
        except _Unreachable:
            return None
        if text is None:
            continue
        found = find_in_sums(text, filename, single_file="{name}" in pattern)
        if found:
            return found, sums_name
    return None
