"""SQLite storage for downloads, their segments and the app settings."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .models import Download, Kind, Segment, Status

_COLUMNS = [f.name for f in fields(Download) if f.name not in ("id", "segments")]
_SENTINEL = object()


class Store:
    def __init__(self, path: Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        # Write-ahead log: progress saves from download threads don't hold up the window.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        cols = ", ".join(_COLUMNS)
        with self._db:
            self._db.execute(f"CREATE TABLE IF NOT EXISTS downloads (id INTEGER PRIMARY KEY, {cols}, segments TEXT)")
            self._db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
            # A download that was running when the app closed is now paused at its saved progress.
            self._db.execute("UPDATE downloads SET status = ? WHERE status = ?", (Status.PAUSED.value, Status.DOWNLOADING.value))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _row_values(d: Download) -> list[Any]:
        values = []
        for name in _COLUMNS:
            v = getattr(d, name)
            values.append(v.value if isinstance(v, (Status, Kind)) else v)
        values.append(json.dumps([asdict(s) for s in d.segments]))
        return values

    @staticmethod
    def _from_row(row: tuple) -> Download:
        data = dict(zip(["id", *_COLUMNS, "segments"], row))
        data["kind"] = Kind(data["kind"])
        data["status"] = Status(data["status"])
        data["resumable"] = bool(data["resumable"])
        data["subtitles"] = bool(data["subtitles"])
        data["segments"] = [Segment(**s) for s in json.loads(data["segments"] or "[]")]
        return Download(**data)

    def add(self, d: Download) -> int:
        return self.add_many([d])[0]

    def add_many(self, downloads: list[Download]) -> list[int]:
        """Insert several downloads in one transaction (a playlist adds hundreds at once)."""
        cols = ", ".join([*_COLUMNS, "segments"])
        marks = ", ".join("?" * (len(_COLUMNS) + 1))
        with self._lock, self._db:
            for d in downloads:
                cur = self._db.execute(f"INSERT INTO downloads ({cols}) VALUES ({marks})", self._row_values(d))
                d.id = cur.lastrowid
        return [d.id for d in downloads]

    def update(self, d: Download) -> None:
        sets = ", ".join(f"{c} = ?" for c in [*_COLUMNS, "segments"])
        with self._lock, self._db:
            self._db.execute(f"UPDATE downloads SET {sets} WHERE id = ?", [*self._row_values(d), d.id])

    def save_segments(self, download_id: int, segments: list[Segment]) -> None:
        """Progress checkpoint; the byte count goes with it so a crash keeps the card's percentage."""
        with self._lock, self._db:
            self._db.execute("UPDATE downloads SET segments = ?, downloaded = ? WHERE id = ?",
                             (json.dumps([asdict(s) for s in segments]), sum(s.done for s in segments),
                              download_id))

    def get(self, download_id: int) -> Download | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM downloads WHERE id = ?", (download_id,)).fetchone()
        return self._from_row(row) if row else None

    def all(self) -> list[Download]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM downloads ORDER BY added_at DESC, id DESC").fetchall()
        return [self._from_row(r) for r in rows]

    def delete(self, download_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM downloads WHERE id = ?", (download_id,))

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, json.dumps(value)))
