"""A tiny line chart of the last minute of total download speed."""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class Sparkline(QWidget):
    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._data: list[float] = []
        self.setFixedSize(84, 24)
        self.setAccessibleName("Download speed over the last minute")

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def set_data(self, data: list[float]) -> None:
        self._data = data
        self.update()

    def paintEvent(self, _event) -> None:
        if len(self._data) < 2:
            return
        top = max(self._data) or 1.0
        w, h = self.width() - 2, self.height() - 4
        step = w / (len(self._data) - 1)
        path = QPainterPath()
        for i, v in enumerate(self._data):
            point = QPointF(1 + i * step, 2 + h - (v / top) * h)
            path.moveTo(point) if i == 0 else path.lineTo(point)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self._color, 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(path)
