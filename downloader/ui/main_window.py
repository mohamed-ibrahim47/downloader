"""The main window, laid out like mockup board B: rail, header, smart bar, cards, queue panel."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from ..engine.linkkind import LinkKind, classify
from ..engine.settings import Settings
from ..engine.store import Store
from .controller import Controller
from .download_list import DownloadList
from .formatting import fmt_speed
from .icons import icon, pixmap
from .queue_panel import QueuePanel
from .smart_bar import SmartBar
from .sparkline import Sparkline
from .theme import PALETTES, stylesheet

CHIPS = (("all", "All"), ("active", "Active"), ("queued", "Queued"), ("completed", "Completed"))
NAV = (("downloads", "list", "Downloads"), ("videos", "video", "Videos and playlists"),
       ("scheduler", "clock", "Scheduler"), ("settings", "sliders", "Settings"))


class MainWindow(QMainWindow):
    def __init__(self, controller: Controller, settings: Settings, store: Store):
        super().__init__()
        self.ctl, self.settings, self.store = controller, settings, store
        self.setWindowTitle("Downloader")
        self.resize(1120, 800)
        self.setMinimumSize(900, 640)
        palette = PALETTES[settings.theme]

        root = QWidget(objectName="root")
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._build_rail())

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_downloads_page(palette))
        self.scheduler_page = QWidget(objectName="page")
        self.settings_page = QWidget(objectName="page")
        self.pages.addWidget(self.scheduler_page)
        self.pages.addWidget(self.settings_page)
        row.addWidget(self.pages, 1)

        self.queue_panel = QueuePanel(settings)
        self.queue_panel.changed.connect(self.apply_settings)
        row.addWidget(self.queue_panel)
        self.setCentralWidget(root)

        self.ctl.changed.connect(self._on_changed)
        self.ctl.removed.connect(self._on_removed)
        self.ctl.speed_tick.connect(self._on_speed)
        self.list.primary_clicked.connect(self.ctl.primary)
        self.list.remove_clicked.connect(self.ctl.remove)

        clipboard = QGuiApplication.clipboard()
        self._last_clip = clipboard.text().strip()  # don't grab whatever was copied before launch
        clipboard.dataChanged.connect(self._on_clipboard)

        self._apply_theme()
        self._update_chips()

    # ---- building -----------------------------------------------------------

    def _build_rail(self) -> QFrame:
        rail = QFrame(objectName="rail")
        rail.setFixedWidth(72)
        col = QVBoxLayout(rail)
        col.setContentsMargins(12, 18, 12, 18)
        col.setSpacing(6)
        self._logo = QLabel(objectName="logo")
        self._logo.setFixedSize(40, 40)
        self._logo.setAlignment(Qt.AlignCenter)
        col.addWidget(self._logo, 0, Qt.AlignHCenter)
        col.addSpacing(14)
        self._nav = QButtonGroup(self)
        self._nav_buttons: dict[str, QToolButton] = {}
        for key, icon_name, label in NAV:
            b = QToolButton(objectName="railButton", checkable=True)
            b.setFixedSize(48, 48)
            b.setIconSize(QSize(22, 22))
            b.setAccessibleName(label)
            b.setToolTip(label)
            b.setProperty("key", key)
            b.setProperty("icon_name", icon_name)
            b.setCursor(Qt.PointingHandCursor)
            self._nav.addButton(b)
            self._nav_buttons[key] = b
            if key == "settings":
                col.addStretch(1)
                col.insertWidget(col.count() - 1, self._build_theme_button(), 0, Qt.AlignHCenter)
            col.addWidget(b, 0, Qt.AlignHCenter)
        self._nav_buttons["downloads"].setChecked(True)
        self._nav.buttonClicked.connect(self._navigate)
        return rail

    def _build_theme_button(self) -> QToolButton:
        self.theme_button = QToolButton(objectName="themeButton")
        self.theme_button.setFixedSize(48, 48)
        self.theme_button.setIconSize(QSize(20, 20))
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.clicked.connect(self._toggle_theme)
        return self.theme_button

    def _build_downloads_page(self, palette) -> QWidget:
        page = QWidget(objectName="page")
        col = QVBoxLayout(page)
        col.setContentsMargins(24, 22, 24, 22)
        col.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(14)
        self.title = QLabel("Downloads", objectName="h1")
        header.addWidget(self.title)
        header.addSpacing(8)
        self._chips = QButtonGroup(self)
        for key, _label in CHIPS:
            chip = QPushButton(objectName="chip", checkable=True)
            chip.setProperty("key", key)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setChecked(key == "all")
            self._chips.addButton(chip)
            header.addWidget(chip)
        self._chips.buttonClicked.connect(lambda b: self.list.set_filter(b.property("key")))
        header.addStretch(1)
        self.sparkline = Sparkline(palette["accent"])
        self.speed_label = QLabel("", objectName="speedTotal")
        header.addWidget(self.sparkline)
        header.addWidget(self.speed_label)
        col.addLayout(header)

        self.smart_bar = SmartBar(self.ctl, palette)
        col.addWidget(self.smart_bar)
        self.list = DownloadList(self.ctl, palette)
        col.addWidget(self.list, 1)
        return page

    # ---- reacting -----------------------------------------------------------

    def _paint_nav(self) -> None:
        p = PALETTES[self.settings.theme]
        for b in self._nav_buttons.values():
            b.setIcon(icon(b.property("icon_name"), p["accent"] if b.isChecked() else p["muted"], 22))

    def _navigate(self, button) -> None:
        self._paint_nav()
        key = button.property("key")
        if key in ("downloads", "videos"):
            self.pages.setCurrentIndex(0)
            self.title.setText("Downloads" if key == "downloads" else "Videos and playlists")
            self.list.set_filter(self.list.mode, youtube_only=key == "videos")
            self._update_chips()
        else:
            self.pages.setCurrentWidget(self.scheduler_page if key == "scheduler" else self.settings_page)

    def _on_changed(self, download_id: int) -> None:
        self.list.update_download(download_id)
        self._update_chips()

    def _on_removed(self, download_id: int) -> None:
        self.list.remove_download(download_id)
        self._update_chips()

    def _on_speed(self, rate: float, history: list) -> None:
        self.sparkline.set_data(history)
        self.speed_label.setText(f"<b>{fmt_speed(rate)}</b> total")

    def _update_chips(self) -> None:
        counts = self.list.counts()
        labels = dict(CHIPS)
        for chip in self._chips.buttons():
            key = chip.property("key")
            chip.setText(f"{labels[key]} {counts[key]}")

    def _on_clipboard(self) -> None:
        text = QGuiApplication.clipboard().text().strip()
        if not self.settings.watch_clipboard or text == self._last_clip:
            return
        self._last_clip = text
        if classify(text) is not LinkKind.INVALID and text != self.smart_bar.input.text().strip():
            self.smart_bar.set_text(text)

    # ---- settings and theme -------------------------------------------------

    def apply_settings(self, settings: Settings) -> None:
        theme_changed = settings.theme != self.settings.theme
        self.settings = settings
        settings.save(self.store)
        self.ctl.apply_settings(settings)
        self.queue_panel.set_settings(settings)
        if theme_changed:
            self._apply_theme()
        else:
            self.smart_bar.set_palette(PALETTES[settings.theme])

    def _toggle_theme(self) -> None:
        import dataclasses
        self.apply_settings(dataclasses.replace(
            self.settings, theme="light" if self.settings.theme == "dark" else "dark"))

    def _apply_theme(self) -> None:
        p = PALETTES[self.settings.theme]
        QApplication.instance().setStyleSheet(stylesheet(self.settings.theme))
        self._logo.setPixmap(pixmap("download", p["onAccent"], 22, 2.4))
        self._paint_nav()
        to_light = self.settings.theme == "dark"
        self.theme_button.setIcon(icon("sun" if to_light else "moon", p["text"], 20))
        label = "Switch to light theme" if to_light else "Switch to dark theme"
        self.theme_button.setAccessibleName(label)
        self.theme_button.setToolTip(label)
        self.sparkline.set_color(p["accent"])
        self.smart_bar.set_palette(p)
        self.list.set_palette(p)
        self.setWindowIcon(icon("download", p["accent"], 64, 2.4))
