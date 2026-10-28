"""Picking part of a video: a bar with two handles over the whole video, and From / To fields that
follow it (or move it when typed in). A Clip switch shows them."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..engine.timecode import fmt_clock, parse_time, start_from_url
from .formatting import fmt_duration
from .range_slider import RangeSlider

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
        self._length = QLabel(objectName="clipLength")
        self.bar = RangeSlider()
        self.bar.range_changed.connect(self._dragged)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("From"))
        row.addWidget(self.from_edit)
        row.addWidget(QLabel("To"))
        row.addWidget(self.to_edit)
        row.addWidget(self._length)
        row.addWidget(self._problem, 1)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        self._bar_start = QLabel("0:00", objectName="clipBarStart")
        self._bar_end = QLabel(objectName="clipBarEnd")
        self._bar_row = QWidget()
        bar_row = QHBoxLayout(self._bar_row)
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.setSpacing(8)
        bar_row.addWidget(self._bar_start)
        bar_row.addWidget(self.bar, 1)
        bar_row.addWidget(self._bar_end)
        col.addWidget(self._bar_row)
        col.addLayout(row)
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
        # The bar needs the video's length; without one (a live stream) only the fields are offered.
        self._bar_row.setVisible(bool(duration))
        if duration:
            self._bar_end.setText(fmt_clock(duration))
            self.bar.set_maximum(duration)
            self.bar.set_values(min(start or 0, duration - 1), duration)
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
        clip = self.clip()
        if clip and self._duration:
            self.bar.set_values(*clip)
        self._refresh()
        self.changed.emit()

    def _dragged(self, start: int, end: int) -> None:
        self.from_edit.setText(fmt_clock(start))
        self.to_edit.setText(fmt_clock(end))

    def _refresh(self) -> None:
        self.setVisible(self.on())
        self._problem.setText(self.problem() or "")
        clip = self.clip()
        self._length.setText(f"{fmt_duration(clip[1] - clip[0])} clip" if clip and clip[1] > clip[0] else "")
