"""A plain-text label that shortens itself with "…" instead of pushing its layout wider."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QSizePolicy

from .formatting import plain_tip


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._full = ""
        self.setTextFormat(Qt.PlainText)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt naming)
        self._full = text or ""
        self.setToolTip(plain_tip(self._full) if self._full else "")
        self._elide()

    def text(self) -> str:
        return self._full

    def shown_text(self) -> str:
        return super().text()

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(20, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = max(self.width(), 0)
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideRight, width) if width else self._full
        super().setText(shown)
