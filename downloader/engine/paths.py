"""Where the app keeps its data, and how downloaded files get safe, unique names."""
from __future__ import annotations

import os
import re
import shutil
import unicodedata
from pathlib import Path

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{dev}{n}" for dev in ("COM", "LPT") for n in "0123456789¹²³"}


def _is_reserved(name: str) -> bool:
    # Windows ignores everything from the first dot, and trailing spaces, when matching devices.
    return name.split(".", 1)[0].rstrip(" ").upper() in _RESERVED
_MAX_NAME = 200


def data_home() -> Path:
    """The app's data folder: $DOWNLOADER_HOME, else %LOCALAPPDATA%\\Downloader."""
    env = os.environ.get("DOWNLOADER_HOME")
    if env:
        home = Path(env)
    else:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
        home = Path(base) / "Downloader"
    home.mkdir(parents=True, exist_ok=True)
    return home


def safe_filename(name: str) -> str:
    """A name Windows accepts and that shows what it is: illegal characters and invisible
    formatting characters (which can disguise `fdp.exe` as `exe.pdf`) removed, reserved device
    names prefixed, trimmed, at most 200 characters."""
    name = "".join(ch for ch in name if unicodedata.category(ch) not in ("Cc", "Cf"))
    name = _ILLEGAL.sub("", name).strip().rstrip(". ").strip()
    if not name:
        return "download"
    if _is_reserved(name):
        name = "_" + name
    if len(name) > _MAX_NAME:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) <= 10:
            name = stem[: _MAX_NAME - len(ext) - 1].rstrip(". ") + "." + ext
        else:
            name = name[:_MAX_NAME]
    return name


def unique_path(folder: Path, name: str) -> Path:
    """`folder/name`, or `name (2).ext`, `name (3).ext`… if it is taken. Never overwrites."""
    folder = Path(folder)
    candidate = folder / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    n = 2
    while True:
        candidate = folder / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def free_space(folder: Path) -> int:
    """Free bytes on the drive holding `folder` (its nearest existing parent)."""
    folder = Path(folder)
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    return shutil.disk_usage(folder).free
