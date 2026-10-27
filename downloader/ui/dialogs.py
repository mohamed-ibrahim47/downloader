"""The full format list for a video (board D) and the pick-your-videos list for a playlist (board C)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton,
                               QVBoxLayout)

from ..engine.youtube import FormatOption, PlaylistInfo, VideoInfo, format_options
from .clip_fields import ClipFields
from .formatting import fmt_bytes, fmt_duration


def _plain(text: str, name: str = "") -> QLabel:
    label = QLabel(text, objectName=name)
    label.setTextFormat(Qt.PlainText)
    return label


def confirm(parent, title: str, text: str, yes_text: str) -> bool:
    """Ask before doing something that can't be undone. The text holds file names from servers,
    so it is shown as plain text, never HTML. No is the default."""
    box = QMessageBox(QMessageBox.Question, title, text, parent=parent)
    box.setTextFormat(Qt.PlainText)
    yes = box.addButton(yes_text, QMessageBox.YesRole)
    box.setDefaultButton(box.addButton("Cancel", QMessageBox.NoRole))
    box.exec()
    return box.clickedButton() is yes


def _buttons(dialog: QDialog, ok_text: str) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addStretch(1)
    cancel = QPushButton("Cancel", objectName="secondaryButton")
    ok = QPushButton(ok_text, objectName="primaryButton")
    ok.setDefault(True)
    cancel.clicked.connect(dialog.reject)
    ok.clicked.connect(dialog.accept)
    row.addWidget(cancel)
    row.addWidget(ok)
    return row


class FormatDialog(QDialog):
    def __init__(self, info: VideoInfo, folder: str, quality: str, subtitles: bool, parent=None,
                 clip_url: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Download video")
        self.setMinimumWidth(560)
        self._options = format_options(info.formats)
        self._group = QButtonGroup(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        layout.addWidget(_plain(info.title, "optTitle"))
        layout.addWidget(_plain(" · ".join(p for p in (info.channel, fmt_duration(info.duration)) if p), "optMeta"))

        wanted = "m4a" if quality == "Audio" else f"v{quality.rstrip('p')}"
        sections = (("Video with sound", [o for o in self._options if o.id.startswith("v")]),
                    ("Sound only", [o for o in self._options if not o.id.startswith("v")]))
        for heading, options in sections:
            if not options:
                continue
            layout.addWidget(QLabel(heading, objectName="panelLabel"))
            for opt in options:
                radio = QRadioButton(f"{opt.quality}    {opt.kind}    {fmt_bytes(opt.size)}")
                radio.setProperty("option_id", opt.id)
                self._group.addButton(radio)
                radio.setChecked(opt.id == wanted)
                layout.addWidget(radio)
        if self._group.checkedButton() is None and self._group.buttons():
            self._group.buttons()[0].setChecked(True)

        self.clip_fields = ClipFields()
        self.clip_fields.set_video(info.duration, clip_url or info.url)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip_fields.toggle)
        clip_row.addWidget(self.clip_fields, 1)
        clip_row.addStretch(0)
        layout.addLayout(clip_row)

        layout.addWidget(QLabel("Save to", objectName="panelLabel"))
        row = QHBoxLayout()
        self._folder = QLineEdit(folder, objectName="field")
        browse = QPushButton("Browse…", objectName="secondaryButton")
        browse.clicked.connect(self._browse)
        row.addWidget(self._folder, 1)
        row.addWidget(browse)
        layout.addLayout(row)

        self._start = QCheckBox("Start right away")
        self._start.setChecked(True)
        self._subs = QCheckBox("Add subtitles if there are any")
        self._subs.setChecked(subtitles)
        checks = QHBoxLayout()
        checks.addWidget(self._start)
        checks.addWidget(self._subs)
        checks.addStretch(1)
        layout.addLayout(checks)
        layout.addLayout(_buttons(self, "Download"))

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Save to", self._folder.text())
        if folder:
            self._folder.setText(folder)

    def accept(self) -> None:
        if self.clip_fields.problem() is None:  # the problem is shown next to the fields
            super().accept()

    def clip(self) -> tuple[int, int] | None:
        return self.clip_fields.clip()

    def selected_option(self) -> FormatOption:
        chosen = self._group.checkedButton().property("option_id")
        return next(o for o in self._options if o.id == chosen)

    def folder(self) -> str:
        return self._folder.text().strip()

    def start_now(self) -> bool:
        return self._start.isChecked()

    def subtitles(self) -> bool:
        return self._subs.isChecked()


class PlaylistDialog(QDialog):
    def __init__(self, info: PlaylistInfo, picked: set[int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose videos")
        self.resize(620, 560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        layout.addWidget(_plain(info.title, "optTitle"))
        self._all = QCheckBox("Select all")
        layout.addWidget(self._all)
        self._list = QListWidget()
        width = 3 if len(info.entries) >= 100 else 2
        for e in info.entries:
            item = QListWidgetItem(f"{e.index:0{width}d}   {e.title}   {fmt_duration(e.duration)}")
            item.setData(Qt.UserRole, e.index)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if e.index in picked else Qt.Unchecked)
            self._list.addItem(item)
        layout.addWidget(self._list, 1)
        layout.addLayout(_buttons(self, "Done"))
        self._all.setChecked(len(picked) == len(info.entries))
        self._all.clicked.connect(self._toggle_all)

    def _toggle_all(self, checked: bool) -> None:
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def picked(self) -> set[int]:
        return {self._list.item(i).data(Qt.UserRole) for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked}
