"""One paste bar for every kind of link: it works out what was pasted and shows the right options."""
from __future__ import annotations

from urllib.parse import urlsplit

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (QButtonGroup, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout, QWidget)

from ..engine.linkkind import LinkKind, classify
from ..engine.probe import FileInfo, probe
from ..engine.youtube import (PlaylistInfo, QUALITIES, VideoInfo, choose_format, clip_size,
                              estimate_playlist_size, fetch_info)
from .clip_fields import ClipFields
from .dialogs import FormatDialog, PlaylistDialog
from .elided import ElidedLabel
from .formatting import fmt_bytes, fmt_duration
from .icons import pixmap
from .workers import run_in_pool

DEBOUNCE_MS = 400
HINT_IDLE = "Copied links are picked up automatically"
HINT_IDLE_NO_WATCH = "A file starts right away; a video or playlist shows its options first"


def _plain(name: str) -> QLabel:
    label = QLabel(objectName=name)
    label.setTextFormat(Qt.PlainText)
    return label


class SmartBar(QWidget):
    def __init__(self, controller, palette: dict[str, str], parent=None):
        super().__init__(parent)
        self.ctl = controller
        self._palette = palette
        self.state = "empty"
        self._token = 0
        self._error = ""
        self._file: FileInfo | None = None
        self._video: VideoInfo | None = None
        self._playlist: PlaylistInfo | None = None
        self._picked: set[int] = set()
        self._quality = "720p"

        frame = QFrame(objectName="smartBar")
        self._link_icon = QLabel()
        self.input = QLineEdit(objectName="smartInput")
        self.input.setPlaceholderText("Paste a link: a file, a YouTube video or a whole playlist")
        self.input.setAccessibleName("Link to download")
        self.input.setMinimumHeight(36)
        self.badge = QLabel(objectName="badge")
        top = QHBoxLayout()
        top.setContentsMargins(14, 6, 8, 6)
        top.setSpacing(10)
        top.addWidget(self._link_icon)
        top.addWidget(self.input, 1)
        top.addWidget(self.badge)

        self.options = QFrame(objectName="smartOptions")
        self._thumb = QLabel(objectName="thumb")
        self._thumb.setFixedSize(80, 46)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._title = ElidedLabel(objectName="optTitle")
        self._meta = ElidedLabel(objectName="optMeta")
        self._more = QPushButton("more formats", objectName="linkButton")
        self._choose = QPushButton("choose videos…", objectName="linkButton")
        for b in (self._more, self._choose):
            b.setCursor(Qt.PointingHandCursor)
        self._more.clicked.connect(self._open_formats)
        self._choose.clicked.connect(self._open_playlist)
        meta_row = QHBoxLayout()
        meta_row.setSpacing(8)
        meta_row.addWidget(self._meta, 1)
        self.clip = ClipFields()
        self.clip.changed.connect(self._render)
        meta_row.addWidget(self._more)
        meta_row.addWidget(self.clip.toggle)
        meta_row.addWidget(self._choose)
        text_col = QVBoxLayout()
        text_col.setSpacing(3)
        text_col.addWidget(self._title)
        text_col.addLayout(meta_row)

        self.quality_group = QFrame(objectName="qualityGroup")
        q_layout = QHBoxLayout(self.quality_group)
        q_layout.setContentsMargins(3, 3, 3, 3)
        q_layout.setSpacing(3)
        self._quality_buttons = QButtonGroup(self)
        for q in QUALITIES:
            b = QPushButton(q, objectName="qualityButton", checkable=True)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(q == self._quality)
            self._quality_buttons.addButton(b)
            q_layout.addWidget(b)
        self._quality_buttons.buttonClicked.connect(self._pick_quality)

        self._size = QLabel(objectName="sizeLabel")
        self._size_note = _plain("sizeNote")
        size_col = QVBoxLayout()
        size_col.setSpacing(2)
        size_col.addWidget(self._size, 0, Qt.AlignRight)
        size_col.addWidget(self._size_note, 0, Qt.AlignRight)
        self.action = QPushButton("Download", objectName="primaryButton")
        self.action.setCursor(Qt.PointingHandCursor)
        self.action.clicked.connect(self._download)

        opt = QHBoxLayout()
        opt.setSpacing(14)
        opt.addWidget(self._thumb)
        opt.addLayout(text_col, 1)
        opt.addWidget(self.quality_group)
        opt.addLayout(size_col)
        opt.addWidget(self.action)
        box = QVBoxLayout(self.options)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(10)
        box.addLayout(opt)
        box.addWidget(self.clip)

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame_layout.addLayout(top)
        frame_layout.addWidget(self.options)

        self.hint = _plain("hint")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(frame)
        outer.addWidget(self.hint)

        self._debounce = QTimer(self, singleShot=True, interval=DEBOUNCE_MS, timeout=self._check)
        self.input.textEdited.connect(lambda _t: self._debounce.start())
        self.input.returnPressed.connect(self._check)
        self._render()

    # ---- public -------------------------------------------------------------

    def set_text(self, url: str) -> None:
        self.input.setText(url)
        self._check()

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self._render()

    # ---- checking a link ----------------------------------------------------

    def _check(self) -> None:
        self._debounce.stop()
        text = self.input.text().strip()
        self._token += 1
        kind = classify(text)
        if not text:
            self._set_state("empty")
            return
        if kind is LinkKind.INVALID:
            self._set_state("invalid")
            return
        self._set_state("checking")
        task = run_in_pool(probe if kind is LinkKind.FILE else fetch_info, text)
        task.setProperty("token", self._token)
        task.done.connect(self._on_result)
        task.failed.connect(self._on_error)

    def _current(self) -> bool:
        task = self.sender()
        return task is None or task.property("token") == self._token

    @Slot(object)
    def _on_result(self, result) -> None:
        if not self._current():
            return
        if isinstance(result, FileInfo):
            self._file = result
            self._set_state("file")
        elif isinstance(result, VideoInfo):
            self._video = result
            self.clip.set_video(result.duration, self.input.text().strip())
            self._set_state("video")
        elif isinstance(result, PlaylistInfo):
            self._playlist = result
            self._picked = {e.index for e in result.entries}
            self._set_state("playlist" if result.entries else "error")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        if not self._current():
            return
        self._error = message
        self._set_state("error")

    # ---- acting -------------------------------------------------------------

    def _pick_quality(self, button) -> None:
        self._quality = button.text()
        self._render()

    def _download(self) -> None:
        if self.state == "file":
            self.ctl.add_file(self._file)
        elif self.state == "video":
            self.ctl.add_video(self._video, choose_format(self._video.formats, self._quality),
                               clip=self.clip.clip())
        elif self.state == "playlist":
            self.ctl.add_playlist(self._playlist, self._quality, sorted(self._picked))
        else:
            return
        self._clear()

    def _clear(self) -> None:
        self._token += 1
        self.input.clear()
        self._set_state("empty")

    def _open_formats(self) -> None:
        s = self.ctl.settings
        dialog = FormatDialog(self._video, s.folder_video, self._quality, s.subtitles, self.window(),
                              clip_url=self.input.text().strip())
        if dialog.exec() == QDialog.Accepted:
            self.ctl.add_video(self._video, dialog.selected_option(), folder=dialog.folder(),
                               subtitles=dialog.subtitles(), start_now=dialog.start_now(), clip=dialog.clip())
            self._clear()

    def _open_playlist(self) -> None:
        dialog = PlaylistDialog(self._playlist, self._picked, self.window())
        if dialog.exec() == QDialog.Accepted:
            self._picked = dialog.picked()
            self._render()

    # ---- drawing ------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        self.state = state
        self._render()

    def _render(self) -> None:
        p, s, state = self._palette, self.ctl.settings, self.state
        self._link_icon.setPixmap(pixmap("link", p["muted"], 18, 2))
        showing = state in ("file", "video", "playlist")
        self.options.setVisible(showing)
        self.badge.setVisible(state != "empty" and state != "invalid")
        hints = {"checking": "Checking link…", "invalid": "That doesn't look like a link", "error": self._error}
        self.hint.setText(hints.get(state, HINT_IDLE if s.watch_clipboard else HINT_IDLE_NO_WATCH))
        self.badge.setText({"checking": "Checking…", "error": "Can't use this link"}.get(state, ""))
        if not showing:
            self.clip.setVisible(False)
            return

        self.quality_group.setVisible(state != "file")
        self._more.setVisible(state == "video")
        self.clip.toggle.setVisible(state == "video")
        self.clip.setVisible(state == "video" and self.clip.on())
        self._choose.setVisible(state == "playlist")
        self.action.setEnabled(True)
        if state == "file":
            f = self._file
            self.badge.setText("File · resumable" if f.resumable else "File · can't resume")
            icon_name, title = "file", f.filename
            meta = f"{urlsplit(f.url).hostname} · saves to {s.folder_general}"
            size = fmt_bytes(f.size) or "Size unknown"
            note = f"{s.connections} connections" if f.resumable else "1 connection"
            action = "Download"
        elif state == "video":
            v = self._video
            self.badge.setText("YouTube video")
            icon_name, title = "video", v.title
            meta = " · ".join(x for x in (v.channel, fmt_duration(v.duration), f"saves to {s.folder_video}") if x)
            choice = choose_format(v.formats, self._quality)
            size = fmt_bytes(choice.size) or "Size unknown"
            note = choice.note or ("M4A sound" if self._quality == "Audio" else "MP4")
            clip = self.clip.clip()
            if clip:
                estimate = clip_size(choice.size, v.duration, *clip)
                size = f"≈ {fmt_bytes(estimate)}" if estimate else "Size unknown"
                note = "clip · estimate"
            self.action.setEnabled(self.clip.problem() is None)
            action = "Download"
        else:
            pl = self._playlist
            n, k = len(pl.entries), len(self._picked)
            self.badge.setText(f"YouTube playlist · {n}")
            icon_name, title = "playlist", pl.title
            total = sum(e.duration or 0 for e in pl.entries)
            meta = " · ".join(x for x in (f"{n} videos", fmt_duration(total)) if x)
            chosen = [e for e in pl.entries if e.index in self._picked]
            size = fmt_bytes(estimate_playlist_size(chosen, self._quality)) or "Size unknown"
            note = f"{k} of {n} videos"
            action = "Download all" if k == n else f"Download {k} videos"
            self.action.setEnabled(k > 0)
        self._thumb.setPixmap(pixmap(icon_name, p["accent"], 20))
        self._title.setText(title)
        self._meta.setText(meta)
        self._size.setText(size)
        self._size_note.setText(note)
        self.action.setText(action)
