"""One download as a card: icon, name, size and source, progress, status, and two buttons."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QToolButton, QVBoxLayout

from ..engine.checksum import Checksum
from ..engine.models import Download, Kind, Status
from ..engine.timecode import clip_label
from .elided import ElidedLabel
from .formatting import fmt_bytes, fmt_eta, fmt_speed, plain_tip
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


def _match_text(expected: Checksum, source: str | None) -> str:
    """Says whose checksum matched: it proves the file is what that source described, no more."""
    whose = f"the {expected.label} you pasted" if source == "pasted" else f"the site's {expected.label}"
    return f"matches {whose} (weak check)" if expected.weak else f"✓ matches {whose}"


def status_texts(d: Download, speed: float | None, eta: float | None) -> tuple[str, str]:
    """The coloured status line and the muted detail line under the progress bar."""
    pct = _pct(d)
    if d.status is Status.DOWNLOADING and d.checking:
        status = "Checking the file…"
        parts = ["comparing with its checksum" if d.checksum else "comparing with the files in its folder"]
    elif d.status is Status.DOWNLOADING:
        if d.downloaded:
            head = f"{pct}%" if pct is not None else fmt_bytes(d.downloaded)
            status = f"{head} · {fmt_speed(speed)}"
            parts = [f"{fmt_bytes(d.downloaded)} done", fmt_eta(eta)]
        else:
            status, parts = "Starting…", ["connecting"]
        if d.kind is Kind.FILE:
            n = d.connections if d.resumable else 1
            parts.append(f"{n} connection{'s' if n != 1 else ''}")
    elif d.status is Status.QUEUED:
        status = "Waiting"
        parts = [f"video {d.playlist_index} in “{d.playlist_title}”" if d.playlist_title else "in the queue"]
    elif d.status is Status.PAUSED:
        status = f"Paused at {pct}%" if pct else "Paused"
        if d.clip:
            parts = ["will start over · clips can't resume"]
        elif d.resumable or d.kind is not Kind.FILE:
            parts = ["resumes where it stopped"]
        else:
            parts = ["will start over · this server can't resume"]
    elif d.status is Status.FAILED:
        status, parts = "Failed", [d.error or "something went wrong"]
    elif d.duplicate_of:
        return "Completed", f"Same as {d.duplicate_of}, which was already in this folder"
    else:
        status, parts = "Completed", [f"saved to {d.folder}"]
        expected = Checksum.from_stored(d.checksum)
        if d.verified and expected:
            parts.append(_match_text(expected, d.checksum_source))
    if d.note:
        parts.append(d.note)
    return status, " · ".join(p for p in parts if p)


class DownloadCard(QFrame):
    primary_clicked = Signal(int)
    remove_clicked = Signal(int)
    delete_copy_clicked = Signal(int)
    keep_both_clicked = Signal(int)

    def __init__(self, download: Download, palette: dict[str, str], parent=None):
        super().__init__(parent)
        self.download = download
        self._palette = palette
        self.setFixedHeight(74)

        self._icon = QLabel(objectName="cardIconBox")
        self._icon.setFixedSize(46, 46)
        self._icon.setAlignment(Qt.AlignCenter)
        self._name = ElidedLabel(objectName="cardName")
        self._meta = QLabel(objectName="cardMeta")
        self._bar = QProgressBar(objectName="cardBar")
        self._bar.setTextVisible(False)
        self._status = QLabel(objectName="cardStatus")
        self._detail = ElidedLabel(objectName="cardDetail")
        for label in (self._meta, self._status):
            label.setTextFormat(Qt.PlainText)
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
        # Offered when the finished file is identical to one already in the folder.
        self._delete_copy = QPushButton("Delete this copy", objectName="linkButton")
        self._keep_both = QPushButton("Keep both", objectName="linkButton")
        for b in (self._delete_copy, self._keep_both):
            b.setCursor(Qt.PointingHandCursor)
        self._delete_copy.clicked.connect(lambda: self.delete_copy_clicked.emit(self.download.id))
        self._keep_both.clicked.connect(lambda: self.keep_both_clicked.emit(self.download.id))
        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        bottom.addWidget(self._status)
        bottom.addWidget(self._detail, 1)
        bottom.addWidget(self._delete_copy)
        bottom.addWidget(self._keep_both)
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
        self._name.setToolTip(plain_tip(d.filename))
        clip = clip_label(*d.clip) if d.clip else None
        self._meta.setText(" · ".join(p for p in (fmt_bytes(d.size), clip, d.quality, d.host) if p))
        status, detail = status_texts(d, speed, eta)
        self._status.setText(status)
        self._detail.setText(detail)
        same = d.status is Status.COMPLETED and bool(d.duplicate_of)
        self._delete_copy.setVisible(same)
        self._keep_both.setVisible(same)
        pct = _pct(d)
        if d.status is Status.DOWNLOADING and (pct is None or d.checking):
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
