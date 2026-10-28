"""A bar with two handles for picking the part of a video to keep (whole seconds)."""
from __future__ import annotations

from PySide6.QtCore import Property, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..engine.timecode import fmt_clock

RADIUS = 8
STEP, BIG_STEP = 1, 10


class RangeSlider(QWidget):
    """Drag either handle, click the bar to bring the nearer handle there, or use ←/→ (Shift: 10 s)
    on the handle last touched. Colours come from the stylesheet (qproperty-accent and friends)."""

    range_changed = Signal(int, int)  # only for the user's own moves, not set_values

    def __init__(self, parent=None):
        super().__init__(parent)
        self._max = 1
        self._start, self._end = 0, 1
        self._active = 1
        self._dragging = False
        self._accent = QColor("#5B8DEF")
        self._track = QColor("#3A3F4B")
        self._handle = QColor("#FFFFFF")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName("Clip range")
        self._describe()

    # ---- values --------------------------------------------------------------

    def maximum(self) -> int:
        return self._max

    def set_maximum(self, seconds: int) -> None:
        self._max = max(1, int(seconds))
        self.set_values(0, self._max)

    def values(self) -> tuple[int, int]:
        return self._start, self._end

    def set_values(self, start: int, end: int) -> None:
        start, end = sorted((int(start), int(end)))
        start = min(max(0, start), self._max - 1)
        end = min(max(start + 1, end), self._max)
        self._start, self._end = start, end
        self._describe()
        self.update()

    def _move(self, handle: int, value: int) -> None:
        if handle == 0:
            value = min(max(0, value), self._end - 1)
            changed = value != self._start
            self._start = value
        else:
            value = max(min(self._max, value), self._start + 1)
            changed = value != self._end
            self._end = value
        if changed:
            self._describe()
            self.update()
            self.range_changed.emit(self._start, self._end)

    def _describe(self) -> None:
        self.setAccessibleDescription(f"from {fmt_clock(self._start)} to {fmt_clock(self._end)}")

    # ---- geometry ------------------------------------------------------------

    def _span(self) -> tuple[float, float]:
        left = RADIUS + 2
        return left, max(left + 1, self.width() - RADIUS - 2)

    def _x(self, value: int) -> float:
        left, right = self._span()
        return left + (right - left) * value / self._max

    def _value_at(self, x: float) -> int:
        left, right = self._span()
        return round((min(max(x, left), right) - left) / (right - left) * self._max)

    def pos_for(self, value: int) -> QPoint:
        return QPoint(round(self._x(value)), self.height() // 2)

    def handle_pos(self, handle: int) -> QPoint:
        return self.pos_for(self._start if handle == 0 else self._end)

    def sizeHint(self) -> QSize:
        return QSize(320, 2 * RADIUS + 12)

    def minimumSizeHint(self) -> QSize:
        return QSize(120, 2 * RADIUS + 12)

    # ---- input ---------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        x = event.position().x()
        d0, d1 = abs(x - self._x(self._start)), abs(x - self._x(self._end))
        # Both handles at one spot: pick by direction, so they can be pulled apart.
        if d0 == d1:
            self._active = 0 if x < self._x(self._start) else 1
        else:
            self._active = 0 if d0 < d1 else 1
        self._dragging = True
        self._move(self._active, self._value_at(x))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._move(self._active, self._value_at(event.position().x()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = False

    def keyPressEvent(self, event) -> None:  # noqa: N802
        step = BIG_STEP if event.modifiers() & Qt.ShiftModifier else STEP
        current = self._start if self._active == 0 else self._end
        if event.key() == Qt.Key_Right:
            self._move(self._active, current + step)
        elif event.key() == Qt.Key_Left:
            self._move(self._active, current - step)
        else:
            super().keyPressEvent(event)

    # ---- drawing -------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        left, right = self._span()
        mid = self.height() / 2
        p.setPen(Qt.NoPen)
        p.setBrush(self._track)
        p.drawRoundedRect(QRectF(left, mid - 2, right - left, 4), 2, 2)
        a, b = self._x(self._start), self._x(self._end)
        p.setBrush(self._accent)
        p.drawRoundedRect(QRectF(a, mid - 3, b - a, 6), 3, 3)
        for i, x in enumerate((a, b)):
            if self.hasFocus() and i == self._active:
                ring = QColor(self._accent)
                ring.setAlpha(70)
                p.setPen(Qt.NoPen)
                p.setBrush(ring)
                p.drawEllipse(QPointF(x, mid), RADIUS + 4, RADIUS + 4)
            p.setPen(QPen(self._accent, 2))
            p.setBrush(self._handle)
            p.drawEllipse(QPointF(x, mid), RADIUS, RADIUS)
        p.end()

    # ---- colours from the stylesheet ------------------------------------------

    def _get_accent(self) -> QColor:
        return self._accent

    def _set_accent(self, c: QColor) -> None:
        self._accent = QColor(c)
        self.update()

    def _get_track(self) -> QColor:
        return self._track

    def _set_track(self, c: QColor) -> None:
        self._track = QColor(c)
        self.update()

    def _get_handle(self) -> QColor:
        return self._handle

    def _set_handle(self, c: QColor) -> None:
        self._handle = QColor(c)
        self.update()

    accent = Property(QColor, _get_accent, _set_accent)
    track = Property(QColor, _get_track, _set_track)
    handle = Property(QColor, _get_handle, _set_handle)
