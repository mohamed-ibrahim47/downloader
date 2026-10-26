"""The right-hand panel: how many at once, connections, theme, clipboard, speed limit, when done."""
from __future__ import annotations

import dataclasses

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..engine.paths import disk_usage
from ..engine.settings import CONNECTION_CHOICES, Settings
from .formatting import fmt_bytes

WHEN_DONE_LABELS = {"nothing": "Do nothing", "notify": "Show a notification", "sleep": "Put the PC to sleep"}


def _label(text: str) -> QLabel:
    return QLabel(text, objectName="panelLabel")


def _segmented(parent, choices: list[tuple[str, object]], current) -> tuple[QHBoxLayout, QButtonGroup]:
    row = QHBoxLayout()
    row.setSpacing(4)
    group = QButtonGroup(parent)
    for text, value in choices:
        b = QPushButton(text, objectName="segButton", checkable=True)
        b.setProperty("value", value)
        b.setChecked(value == current)
        b.setCursor(Qt.PointingHandCursor)
        group.addButton(b)
        row.addWidget(b)
    return row, group


class QueuePanel(QFrame):
    """The Queue page (left menu): how downloads share the connection, and what happens at the end."""

    changed = Signal(object)  # Settings

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent, objectName="page")
        self._settings = settings
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 22)
        outer.setSpacing(18)
        outer.addWidget(QLabel("Queue", objectName="h1"))
        intro = QLabel("How many downloads run together, how fast, and what happens when they're done.",
                       objectName="muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        column = QWidget()
        column.setMaximumWidth(420)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(18)
        outer.addWidget(column, 1)

        box = QVBoxLayout()
        box.setSpacing(8)
        box.addWidget(_label("Downloads at the same time"))
        step = QHBoxLayout()
        step.setSpacing(8)
        minus = QPushButton("−", objectName="stepButton")
        minus.setAccessibleName("Fewer")
        plus = QPushButton("+", objectName="stepButton")
        plus.setAccessibleName("More")
        self._count = QLabel(str(settings.max_active), objectName="stepValue")
        self._count.setFixedWidth(40)
        self._count.setAlignment(Qt.AlignCenter)
        minus.clicked.connect(lambda: self._step(-1))
        plus.clicked.connect(lambda: self._step(+1))
        step.addWidget(minus)
        step.addWidget(self._count)
        step.addWidget(plus)
        step.addStretch(1)
        box.addLayout(step)
        layout.addLayout(box)

        box = QVBoxLayout()
        box.setSpacing(8)
        box.addWidget(_label("Connections per file"))
        row, self._conns = _segmented(self, [(str(n), n) for n in CONNECTION_CHOICES], settings.connections)
        self._conns.buttonClicked.connect(lambda b: self._set(connections=b.property("value")))
        box.addLayout(row)
        layout.addLayout(box)

        self._clip = QCheckBox("Watch the clipboard")
        self._clip.setToolTip("Offer to download links you copy")
        self._clip.setChecked(settings.watch_clipboard)
        self._clip.toggled.connect(lambda on: self._set(watch_clipboard=on))
        layout.addWidget(self._clip)

        limit = QVBoxLayout()
        limit.setSpacing(8)
        self._limit = QCheckBox("Limit speed")
        self._limit.setToolTip("Keep bandwidth for other apps")
        self._limit.setChecked(settings.speed_limit_on)
        self._kbps = QSpinBox()
        self._kbps.setRange(16, 1_000_000)
        self._kbps.setSingleStep(256)
        self._kbps.setSuffix(" KB/s")
        self._kbps.setValue(settings.speed_limit_kbps)
        self._kbps.setVisible(settings.speed_limit_on)
        self._limit.toggled.connect(self._on_limit)
        self._kbps.valueChanged.connect(lambda v: self._set(speed_limit_kbps=v))
        limit.addWidget(self._limit)
        limit.addWidget(self._kbps)
        layout.addLayout(limit)

        box = QVBoxLayout()
        box.setSpacing(8)
        box.addWidget(_label("When the queue finishes"))
        self._when = QComboBox()
        self._when.setAccessibleName("When the queue finishes")
        for key, text in WHEN_DONE_LABELS.items():
            self._when.addItem(text, key)
        self._when.setCurrentIndex(list(WHEN_DONE_LABELS).index(settings.when_done))
        self._when.currentIndexChanged.connect(lambda _i: self._set(when_done=self._when.currentData()))
        box.addWidget(self._when)
        layout.addLayout(box)

        layout.addStretch(1)
        disk = QVBoxLayout()
        disk.setSpacing(6)
        self._saving = _label("")
        self._saving.setTextFormat(Qt.PlainText)
        self._disk = QProgressBar(objectName="diskBar")
        self._disk.setTextVisible(False)
        self._free = _label("")
        disk.addWidget(self._saving)
        disk.addWidget(self._disk)
        disk.addWidget(self._free)
        layout.addLayout(disk)
        self._refresh_disk()
        self._disk_timer = QTimer(self, interval=30_000, timeout=self._refresh_disk)
        self._disk_timer.start()

    def set_settings(self, settings: Settings) -> None:
        """Reflect settings changed elsewhere (the rail's theme button, the Settings page)."""
        self._settings = settings
        for b in self._conns.buttons():
            b.setChecked(b.property("value") == settings.connections)
        self._count.setText(str(settings.max_active))
        self._refresh_disk()

    def _set(self, **changes) -> None:
        self._settings = dataclasses.replace(self._settings, **changes)
        self.changed.emit(self._settings)

    def _step(self, delta: int) -> None:
        n = min(10, max(1, self._settings.max_active + delta))
        self._count.setText(str(n))
        self._set(max_active=n)

    def _on_limit(self, on: bool) -> None:
        self._kbps.setVisible(on)
        self._set(speed_limit_on=on)

    def _refresh_disk(self) -> None:
        folder = self._settings.folder_general
        self._saving.setText(f"Saving to {folder}")
        try:
            usage = disk_usage(folder)
        except OSError:
            self._free.setText("")
            return
        self._disk.setRange(0, 1000)
        self._disk.setValue(int(usage.used / usage.total * 1000) if usage.total else 0)
        self._free.setText(f"{fmt_bytes(usage.free)} free")
