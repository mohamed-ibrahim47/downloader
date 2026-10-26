"""The window's own title bar, drawn in the theme's colours instead of the Windows default.

Dragging, double-click to maximise and snapping to screen edges go through Qt's
startSystemMove, so Windows still handles them natively.
"""
from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QWidget

from .icons import icon, pixmap

GRIP = 6  # px along the border where dragging resizes the window

_CURSORS = {
    Qt.LeftEdge: Qt.SizeHorCursor, Qt.RightEdge: Qt.SizeHorCursor,
    Qt.TopEdge: Qt.SizeVerCursor, Qt.BottomEdge: Qt.SizeVerCursor,
    Qt.LeftEdge | Qt.TopEdge: Qt.SizeFDiagCursor, Qt.RightEdge | Qt.BottomEdge: Qt.SizeFDiagCursor,
    Qt.RightEdge | Qt.TopEdge: Qt.SizeBDiagCursor, Qt.LeftEdge | Qt.BottomEdge: Qt.SizeBDiagCursor,
}


class EdgeGrip(QWidget):
    """An invisible strip on one side or corner of the frameless window; dragging it resizes."""

    def __init__(self, window, edges):
        super().__init__(window)
        self.edges = edges
        self.setCursor(_CURSORS[edges])

    def mousePressEvent(self, event) -> None:  # noqa: N802
        handle = self.window().windowHandle()
        if event.button() == Qt.LeftButton and handle is not None:
            handle.startSystemResize(self.edges)
            event.accept()


def make_grips(window) -> list[EdgeGrip]:
    return [EdgeGrip(window, edges) for edges in _CURSORS]


def place_grips(window, grips: list[EdgeGrip]) -> None:
    """Lay the grips along the window's border (call on resize); hide them when maximised."""
    w, h, g = window.width(), window.height(), GRIP
    rects = {
        Qt.LeftEdge: QRect(0, g, g, h - 2 * g), Qt.RightEdge: QRect(w - g, g, g, h - 2 * g),
        Qt.TopEdge: QRect(g, 0, w - 2 * g, g), Qt.BottomEdge: QRect(g, h - g, w - 2 * g, g),
        Qt.LeftEdge | Qt.TopEdge: QRect(0, 0, g, g), Qt.RightEdge | Qt.TopEdge: QRect(w - g, 0, g, g),
        Qt.LeftEdge | Qt.BottomEdge: QRect(0, h - g, g, g), Qt.RightEdge | Qt.BottomEdge: QRect(w - g, h - g, g, g),
    }
    resizable = not window.isMaximized()
    for grip in grips:
        grip.setGeometry(rects[grip.edges])
        grip.setVisible(resizable)
        grip.raise_()


class TitleBar(QFrame):
    def __init__(self, window, title: str, palette: dict[str, str]):
        super().__init__(window, objectName="titleBar")
        self._window = window
        self._palette = palette
        self.setFixedHeight(36)

        self._icon = QLabel()
        self.title = QLabel(title, objectName="appTitle")
        self._min = self._button("minButton", "Minimise", window.showMinimized)
        self._max = self._button("maxButton", "Maximise", self.toggle_maximised)
        self._close = self._button("closeButton", "Close", window.close)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._icon)
        row.addSpacing(10)
        row.addWidget(self.title)
        row.addStretch(1)
        for b in (self._min, self._max, self._close):
            row.addWidget(b)
        self.set_palette(palette)

    def _button(self, name: str, label: str, slot) -> QToolButton:
        b = QToolButton(objectName=name)
        b.setFixedSize(46, 36)
        b.setIconSize(QSize(14, 14))
        b.setAccessibleName(label)
        b.setToolTip(label)
        b.clicked.connect(slot)
        return b

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        self._icon.setPixmap(pixmap("download", palette["accent"], 16, 2.4))
        self._paint_buttons()

    def _paint_buttons(self) -> None:
        color = self._palette["text"]
        maximised = self._window.isMaximized()
        self._min.setIcon(icon("win_min", color, 14, 1.4))
        self._max.setIcon(icon("win_restore" if maximised else "win_max", color, 14, 1.4))
        self._max.setAccessibleName("Restore" if maximised else "Maximise")
        self._max.setToolTip(self._max.accessibleName())
        self._close.setIcon(icon("win_close", color, 14, 1.4))

    def toggle_maximised(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def window_state_changed(self) -> None:
        self._paint_buttons()

    # ---- moving the window --------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._window.windowHandle() is not None:
            self._window.windowHandle().startSystemMove()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.toggle_maximised()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
