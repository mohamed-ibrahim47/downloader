"""The scrolling list of download cards, its filters, and the folded "Completed today" row."""
from __future__ import annotations

import time
from datetime import date, datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from ..engine.models import Download, Kind, Status
from .card import DownloadCard
from .formatting import fmt_bytes
from .icons import icon

FILTERS = ("all", "active", "queued", "completed")

HIGHLIGHT_MS = 1500


def _today(d: Download) -> bool:
    return bool(d.finished_at) and datetime.fromtimestamp(d.finished_at).date() == date.today()


class DownloadList(QScrollArea):
    primary_clicked = Signal(int)
    remove_clicked = Signal(int)
    delete_copy_clicked = Signal(int)
    keep_both_clicked = Signal(int)

    def __init__(self, controller, palette: dict[str, str], parent=None):
        super().__init__(parent)
        self.ctl = controller
        self._palette = palette
        self._cards: dict[int, DownloadCard] = {}
        self.mode = "all"
        self.youtube_only = False
        self.completed_open = False

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget(objectName="page")
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(0, 0, 4, 0)
        self._layout.setSpacing(8)
        self._active_box = QVBoxLayout()
        self._active_box.setSpacing(8)
        self._done_box = QVBoxLayout()
        self._done_box.setSpacing(8)
        self.completed_row = QPushButton(objectName="completedRow")
        self.completed_row.setCursor(Qt.PointingHandCursor)
        self.completed_row.clicked.connect(self._toggle_completed)
        self._empty = QLabel("Nothing here yet. Paste a link above to start.", objectName="muted")
        self._empty.setAlignment(Qt.AlignCenter)
        self._layout.addLayout(self._active_box)
        self._layout.addWidget(self.completed_row)
        self._layout.addLayout(self._done_box)
        self._layout.addWidget(self._empty)
        self._layout.addStretch(1)
        self.setWidget(body)

        for d in self.ctl.downloads():
            self._add_card(d)
        self.refresh()

    # ---- cards --------------------------------------------------------------

    def _add_card(self, d: Download) -> DownloadCard:
        card = DownloadCard(d, self._palette)
        card.primary_clicked.connect(self.primary_clicked)
        card.remove_clicked.connect(self.remove_clicked)
        card.delete_copy_clicked.connect(self.delete_copy_clicked)
        card.keep_both_clicked.connect(self.keep_both_clicked)
        self._cards[d.id] = card
        return card

    def update_download(self, download_id: int) -> None:
        d = self.ctl.get(download_id)
        if d is None:
            return
        card = self._cards.get(download_id) or self._add_card(d)
        before = card.property("status")
        card.update_from(d, *self.ctl.speed_of(download_id))
        if before != d.status.value or card.parent() is None:
            self.refresh()

    def reveal(self, download_id: int) -> bool:
        """Scroll to a card and outline it briefly. False if the current filter hides it."""
        card = self._cards.get(download_id)
        if card is None or card.parent() is None or card.isHidden():
            return False
        self.ensureWidgetVisible(card)
        self._set_highlight(card, True)
        QTimer.singleShot(HIGHLIGHT_MS, lambda: self._set_highlight(card, False))
        return True

    @staticmethod
    def _set_highlight(card: DownloadCard, on: bool) -> None:
        try:
            card.setProperty("highlight", on)
            card.style().unpolish(card)
            card.style().polish(card)
        except RuntimeError:
            pass  # the card was removed meanwhile

    def remove_download(self, download_id: int) -> None:
        card = self._cards.pop(download_id, None)
        if card is not None:
            card.setParent(None)
            card.deleteLater()
            self.refresh()

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = palette
        for card in self._cards.values():
            card.set_palette(palette)
        self._paint_row()

    # ---- filtering ----------------------------------------------------------

    def _in_view(self, d: Download) -> bool:
        if self.youtube_only and d.kind is Kind.FILE:
            return False
        return True

    def counts(self) -> dict[str, int]:
        ds = [d for d in self.ctl.downloads() if self._in_view(d)]
        return {
            "all": len(ds),
            "active": sum(d.status is Status.DOWNLOADING for d in ds),
            "queued": sum(d.status is Status.QUEUED for d in ds),
            "completed": sum(d.status is Status.COMPLETED for d in ds),
        }

    def set_filter(self, mode: str, youtube_only: bool | None = None) -> None:
        self.mode = mode
        if youtube_only is not None:
            self.youtube_only = youtube_only
        self.refresh()

    def _toggle_completed(self) -> None:
        self.completed_open = not self.completed_open
        self.refresh()

    def refresh(self) -> None:
        """Put each card in the right place and show only those the current filter wants."""
        ds = [d for d in self.ctl.downloads() if self._in_view(d)]
        active = [d for d in ds if d.status is not Status.COMPLETED]
        done = sorted((d for d in ds if d.status is Status.COMPLETED),
                      key=lambda d: d.finished_at or 0, reverse=True)
        if self.mode == "active":
            active, done = [d for d in active if d.status is Status.DOWNLOADING], []
        elif self.mode == "queued":
            active, done = [d for d in active if d.status is Status.QUEUED], []
        elif self.mode == "completed":
            active = []

        today = [d for d in done if _today(d)]
        fold = self.mode == "all"
        shown_done = (today if self.completed_open else []) if fold else done
        shown = {d.id for d in active} | {d.id for d in shown_done}

        for box, items in ((self._active_box, active), (self._done_box, shown_done)):
            for i, d in enumerate(items):
                card = self._cards.get(d.id) or self._add_card(d)
                box.insertWidget(i, card)
        for download_id, card in self._cards.items():
            card.setVisible(download_id in shown)

        self.completed_row.setVisible(fold and bool(today))
        n, size = len(today), sum(d.size or d.downloaded for d in today)
        self._row_text = f"Completed today · {n} file{'s' if n != 1 else ''} · {fmt_bytes(size)}"
        self._paint_row()
        self._empty.setVisible(not shown and not self.completed_row.isVisible())

    def _paint_row(self) -> None:
        label = "Hide" if self.completed_open else "Show"
        self.completed_row.setText(f"{self._row_text}    ·    {label}")
        self.completed_row.setIcon(icon("check", self._palette["done"], 16, 2.2))
