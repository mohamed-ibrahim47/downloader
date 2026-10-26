"""One download as a card: icon, name, size and source, progress, status, and two buttons."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QToolButton, QVBoxLayout

from ..engine.models import Download, Kind, Status
from .formatting import fmt_bytes, fmt_eta, fmt_speed
from .icons import icon, pixmap
from .theme import status_color

_ARCHIVES = {".zip", ".rar", ".7z", ".iso", ".gz", ".tar", ".xz", ".bz2", ".img"}
_AUDIO = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus", ".aac"}
_VIDEO = {".mp4", ".mkv", ".webm", ".avi", ".mov"}

_PRIMARY = {
    Status.DOWNLOADING: ("pause", "Pause"),
    Status.QUEUED: ("play", "Start now"),
    Status.PAUSED: ("play", "Resume"),
    Status.FAILED: ("retry", "Retry"),
    Status.COMPLETED: ("folder", "Open folder"),
}


def kind_icon(d: Download) -> str:
    ext = Path(d.filename).suffix.lower()
    if d.kind in (Kind.VIDEO, Kind.PLAYLIST_ITEM):
        return "music" if (d.quality == "Audio" or ext in _AUDIO) else "video"
    if ext in _ARCHIVES:
        return "archive"
    if ext in _AUDIO:
        return "music"
    if ext in _VIDEO:
        return "video"
    return "file"


def _pct(d: Download) -> int | None:
    if not d.size:
        return None
    return min(100, int(d.downloaded * 100 / d.size))


def status_texts(d: Download, speed: float | None, eta: float | None) -> tuple[str, str]:
    """The coloured status line and the muted detail line under the progress bar."""
    pct = _pct(d)
    if d.status is Status.DOWNLOADING:
        head = f"{pct}%" if pct is not None else fmt_bytes(d.downloaded)
        status = f"{head} · {fmt_speed(speed)}"
        parts = [f"{fmt_bytes(d.downloaded)} done", fmt_eta(eta)]
        if d.kind is Kind.FILE:
            n = d.connections if d.resumable else 1
            parts.append(f"{n} connection{'s' if n != 1 else ''}")
    elif d.status is Status.QUEUED:
        status = "Waiting"
        parts = [f"video {d.playlist_index} in “{d.playlist_title}”" if d.playlist_title else "in the queue"]
    elif d.status is Status.PAUSED:
        status = f"Paused at {pct}%" if pct else "Paused"
        resumable = d.resumable or d.kind is not Kind.FILE
        parts = ["resumes where it stopped" if resumable else "will start over · this server can't resume"]
    elif d.status is Status.FAILED:
        status, parts = "Failed", [d.error or "something went wrong"]
    else:
        status, parts = "Completed", [f"saved to {d.folder}"]
    if d.note:
        parts.append(d.note)
    return status, " · ".join(p for p in parts if p)


class DownloadCard(QFrame):
    primary_clicked = Signal(int)
    remove_clicked = Signal(int)

    def __init__(self, download: Download, palette: dict[str, str], parent=None):
        super().__init__(parent)
        self.download = download
        self._palette = palette
        self.setFixedHeight(74)

        self._icon = QLabel(objectName="cardIconBox")
        self._icon.setFixedSize(46, 46)
        self._icon.setAlignment(Qt.AlignCenter)
        self._name = QLabel(objectName="cardName")
        self._name.setTextFormat(Qt.PlainText)
        self._meta = QLabel(objectName="cardMeta")
        self._bar = QProgressBar(objectName="cardBar")
        self._bar.setTextVisible(False)
        self._status = QLabel(objectName="cardStatus")
        self._detail = QLabel(objectName="cardDetail")
        self._detail.setTextFormat(Qt.PlainText)
        self._primary = QToolButton(objectName="cardPrimary")
        self._remove = QToolButton(objectName="cardRemove")
        self._remove.setAccessibleName("Remove")
        self._remove.setToolTip("Remove")
        for b in (self._primary, self._remove):
            b.setFixedSize(36, 36)
            b.setIconSize(QSize(16, 16))
            b.setCursor(Qt.PointingHandCursor)
        self._primary.clicked.connect(lambda: self.primary_clicked.emit(self.download.id))
        self._remove.clicked.connect(lambda: self.remove_clicked.emit(self.download.id))

        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(self._name, 1)
        top.addWidget(self._meta, 0, Qt.AlignRight)
        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        bottom.addWidget(self._status)
        bottom.addWidget(self._detail, 1)
        middle = QVBoxLayout()
        middle.setSpacing(6)
        middle.addLayout(top)
        middle.addWidget(self._bar)
        middle.addLayout(bottom)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(14)
        row.addWidget(self._icon)
        row.addLayout(middle, 1)
        row.addWidget(self._primary)
        row.addWidget(self._remove)

        self.update_from(download)

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self._paint_icons()

    def update_from(self, d: Download, speed: float | None = None, eta: float | None = None) -> None:
        old_status = self.property("status")
        self.download = d
        self._name.setText(d.title if d.kind is not Kind.FILE and d.title and d.status is not Status.COMPLETED
                           else d.filename)
        self._name.setToolTip(d.filename)
        self._meta.setText(" · ".join(p for p in (fmt_bytes(d.size), d.quality, d.host) if p))
        status, detail = status_texts(d, speed, eta)
        self._status.setText(status)
        self._detail.setText(detail)
        self._detail.setToolTip(detail)
        pct = _pct(d)
        if d.status is Status.DOWNLOADING and pct is None:
            self._bar.setRange(0, 0)
        else:
            self._bar.setRange(0, 100)
            self._bar.setValue(100 if d.status is Status.COMPLETED else (pct or 0))
        name, label = _PRIMARY[d.status]
        self._primary.setAccessibleName(label)
        self._primary.setToolTip(label)
        self._primary_icon = name
        if old_status != d.status.value:
            self.setProperty("status", d.status.value)
            for w in (self, self._status, self._bar):
                w.style().unpolish(w)
                w.style().polish(w)
            self._paint_icons()

    def _paint_icons(self) -> None:
        p = self._palette
        tint = status_color(p, self.download.status.value)
        self._icon.setPixmap(pixmap(kind_icon(self.download), tint, 22))
        self._primary.setIcon(icon(self._primary_icon, p["text"], 16, 2))
        self._remove.setIcon(icon("x", p["muted"], 16, 2))
