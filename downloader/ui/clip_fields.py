"""From / To fields for downloading just part of a video, with a Clip switch that shows them."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from ..engine.timecode import fmt_clock, parse_time, start_from_url
from .formatting import fmt_duration

BAD_TIME = "use a time like 12:30 or 1:02:03"


class ClipFields(QWidget):
    """The row of fields. `toggle` is the Clip switch; the host places it where it fits."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration: int | None = None
        self.toggle = QPushButton("Clip", objectName="clipToggle", checkable=True)
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.setToolTip("Download just part of the video")
        self.from_edit = QLineEdit(objectName="clipFrom")
        self.to_edit = QLineEdit(objectName="clipTo")
        for edit, name in ((self.from_edit, "From"), (self.to_edit, "To")):
            edit.setAccessibleName(f"Clip {name.lower()}")
            edit.setProperty("role", "field")
            edit.setFixedWidth(84)
            edit.textChanged.connect(self._edited)
        self.from_edit.setPlaceholderText("start")
        self.to_edit.setPlaceholderText("end")
        self._problem = QLabel(objectName="clipProblem")
        self._problem.setTextFormat(Qt.PlainText)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(QLabel("From"))
        row.addWidget(self.from_edit)
        row.addWidget(QLabel("To"))
        row.addWidget(self.to_edit)
        row.addWidget(self._problem, 1)
        self.toggle.toggled.connect(self._toggled)
        self.setVisible(False)

    def set_video(self, duration: int | None, url: str) -> None:
        """A new video: switch off, From from the link's start time, To at the end."""
        self._duration = duration
        start = start_from_url(url)
        self.blockSignals(True)
        self.toggle.setChecked(False)
        self.from_edit.setText(fmt_clock(start) if start is not None else "")
        self.to_edit.setText(fmt_clock(duration) if duration else "")
        self.blockSignals(False)
        self._refresh()

    def on(self) -> bool:
        return self.toggle.isChecked()

    def problem(self) -> str | None:
        if not self.on():
            return None
        start_text, end_text = self.from_edit.text().strip(), self.to_edit.text().strip()
        start = parse_time(start_text) if start_text else 0
        end = parse_time(end_text) if end_text else self._duration
        if start is None or (end_text and end is None):
            return BAD_TIME
        if end is None:
            return "enter where the clip ends"
        if start >= end:
            return "From must be before To"
        if self._duration and end > self._duration:
            return f"the video is {fmt_duration(self._duration)} long"
        return None

    def clip(self) -> tuple[int, int] | None:
        if not self.on() or self.problem():
            return None
        end_text = self.to_edit.text().strip()
        start_text = self.from_edit.text().strip()
        return (parse_time(start_text) if start_text else 0,
                parse_time(end_text) if end_text else self._duration)

    def _toggled(self, _checked: bool) -> None:
        self._refresh()
        self.changed.emit()

    def _edited(self, _text: str) -> None:
        self._refresh()
        self.changed.emit()

    def _refresh(self) -> None:
        self.setVisible(self.on())
        self._problem.setText(self.problem() or "")
