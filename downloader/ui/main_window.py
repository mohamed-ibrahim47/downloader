"""The main window, laid out like mockup board B: rail, header, smart bar, cards, queue panel."""
from __future__ import annotations

import dataclasses
from datetime import datetime

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QStackedWidget, QSystemTrayIcon, QToolButton, QVBoxLayout,
                               QWidget)

from ..engine.errors import CHECKSUM_MISMATCH
from ..engine.linkkind import LinkKind, classify
from ..engine.models import Kind, Status
from ..engine.power import sleep_pc
from ..engine.scheduler import due_action
from ..engine.settings import Settings
from ..engine.store import Store
from .controller import Controller
from .dialogs import confirm
from .download_list import DownloadList
from .formatting import fmt_speed
from .icons import icon, pixmap
from .pages import SchedulerPage, SettingsPage
from .queue_panel import QueuePanel
from .smart_bar import SmartBar
from .sparkline import Sparkline
from .theme import APP_ICON, PALETTES, stylesheet
from .title_bar import TitleBar, make_grips, place_grips
from .tray import Tray

CHIPS = (("all", "All"), ("active", "Active"), ("queued", "Queued"), ("completed", "Completed"))
NAV = (("downloads", "list", "Downloads"), ("videos", "video", "Videos and playlists"),
       ("queue", "gauge", "Queue"), ("scheduler", "clock", "Scheduler"), ("settings", "sliders", "Settings"))


class MainWindow(QMainWindow):
    def __init__(self, controller: Controller, settings: Settings, store: Store):
        super().__init__()
        self.ctl, self.settings, self.store = controller, settings, store
        # The app draws its own title bar in the theme's colours (Windows' own can't follow it).
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint
                            | Qt.WindowSystemMenuHint)
        self.setWindowTitle("Downloader")
        self.resize(1120, 800)
        self.setMinimumSize(900, 640)
        palette = PALETTES[settings.theme]

        root = QWidget(objectName="root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.title_bar = TitleBar(self, "Downloader", palette)
        outer.addWidget(self.title_bar)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)
        body.addWidget(self._build_rail())

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_downloads_page(palette))
        self.queue_panel = QueuePanel(settings)
        self.queue_panel.changed.connect(self.apply_settings)
        self.queue_page = self.queue_panel
        self.scheduler_page = SchedulerPage(settings, self.apply_settings)
        self.settings_page = SettingsPage(settings, self.apply_settings)
        for page in (self.queue_page, self.scheduler_page, self.settings_page):
            self.pages.addWidget(page)
        body.addWidget(self.pages, 1)
        self.setCentralWidget(root)
        self._grips = make_grips(self)  # resizing from the border, since there's no system frame

        self.ctl.changed.connect(self._on_changed)
        self.ctl.removed.connect(self._on_removed)
        self.ctl.speed_tick.connect(self._on_speed)
        self.ctl.finished.connect(self._on_finished)
        self.ctl.queue_finished.connect(self._on_queue_finished)

        self.tray = None
        self.sleep_pc = sleep_pc
        self._quitting = False
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = Tray(self, self.ctl, palette["accent"])
            self.tray.show()
        self._sched_prev = datetime.now()
        self._sched_timer = QTimer(self, interval=30_000, timeout=self._scheduler_tick)
        self._sched_timer.start()
        self.list.primary_clicked.connect(self._on_primary)
        self.list.remove_clicked.connect(self.ctl.remove)
        self.list.delete_copy_clicked.connect(self._on_delete_copy)
        self.list.keep_both_clicked.connect(self.ctl.keep_both)
        self.smart_bar.show_requested.connect(self.show_download)

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
        self.speed_label.setToolTip("Total download speed")
        self.speed_label.setMinimumWidth(72)
        self.speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
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
            pages = {"queue": self.queue_page, "scheduler": self.scheduler_page, "settings": self.settings_page}
            self.pages.setCurrentWidget(pages[key])

    def _on_primary(self, download_id: int) -> None:
        d = self.ctl.get(download_id)
        damaged = (d is not None and d.status is Status.FAILED and d.error == CHECKSUM_MISMATCH
                   and d.part_path.exists())
        if not damaged:
            self.ctl.primary(download_id)
        elif confirm(self, "Download again", f"Delete the damaged {d.filename} and download it again?",
                     "Delete and download"):
            self.ctl.redownload(download_id)

    def _on_delete_copy(self, download_id: int) -> None:
        d = self.ctl.get(download_id)
        if d is not None and confirm(
                self, "Delete this copy",
                f"Delete {d.filename}? It's the same as {d.duplicate_of}, which stays in the folder.",
                "Delete this copy"):
            self.ctl.delete_copy(download_id)

    def show_download(self, download_id: int) -> None:
        """Bring a download's card into view, clearing a filter that hides it."""
        if self.list.reveal(download_id):
            return
        d = self.ctl.get(download_id)
        if d is not None and d.kind is Kind.FILE and self.list.youtube_only:
            self._nav_buttons["downloads"].setChecked(True)
            self._navigate(self._nav_buttons["downloads"])
        for chip in self._chips.buttons():
            chip.setChecked(chip.property("key") == "all")
        self.list.set_filter("all")
        self.list.reveal(download_id)

    def _on_changed(self, download_id: int) -> None:
        self.list.update_download(download_id)
        self._update_chips()

    def _on_removed(self, download_id: int) -> None:
        self.list.remove_download(download_id)
        self._update_chips()

    def _on_speed(self, rate: float, history: list) -> None:
        self.sparkline.set_data(history)
        self.speed_label.setText(f"<b>{fmt_speed(rate)}</b>")

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

    # ---- finishing, scheduling, tray ---------------------------------------

    def _notify(self, title: str, text: str) -> None:
        if self.tray is not None:
            self.tray.notify(title, text)

    def _on_finished(self, download_id: int) -> None:
        d = self.ctl.get(download_id)
        if d is None or self.isActiveWindow():
            return  # you're looking at the card already
        if d.status is Status.FAILED:
            self._notify(f"{d.filename} failed", d.error or "")
        else:
            self._notify(f"{d.filename} finished", f"Saved to {d.folder}")

    def _on_queue_finished(self) -> None:
        if self.settings.when_done == "notify":
            self._notify("All downloads finished", "Your queue is empty.")
        elif self.settings.when_done == "sleep":
            self._notify("All downloads finished", "Putting the PC to sleep.")
            self.sleep_pc()

    def _scheduler_tick(self, now: datetime | None = None) -> None:
        now = now or datetime.now()
        action = due_action(self.settings.schedule, self._sched_prev, now)
        self._sched_prev = now
        if action == "start":
            self.ctl.start_queue()
        elif action == "stop":
            self.ctl.stop_queue()

    # ---- frameless window: resize grips along the border --------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        place_grips(self, self._grips)

    def changeEvent(self, event) -> None:  # noqa: N802
        if event.type() == QEvent.WindowStateChange:
            self.title_bar.window_state_changed()
            place_grips(self, self._grips)
        super().changeEvent(event)

    def bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def quit_app(self) -> None:
        self._quitting = True
        self.close()
        QApplication.instance().quit()

    def closeEvent(self, event) -> None:  # noqa: N802
        busy = any(d.status is Status.DOWNLOADING for d in self.ctl.downloads())
        if busy and self.tray is not None and not self._quitting:
            event.ignore()
            self.hide()
            self._notify("Still downloading", "Downloader keeps going here. Right-click to quit.")
            return
        event.accept()
        app = QApplication.instance()
        if not app.quitOnLastWindowClosed():  # the tray keeps the app alive; nothing to wait for
            app.quit()

    # ---- settings and theme -------------------------------------------------

    def apply_settings(self, settings: Settings) -> None:
        theme_changed = settings.theme != self.settings.theme
        self.settings = settings
        settings.save(self.store)
        self.ctl.apply_settings(settings)
        self.queue_panel.set_settings(settings)
        self.settings_page.set_settings(settings)
        if theme_changed:
            self._apply_theme()
        else:
            self.smart_bar.set_palette(PALETTES[settings.theme])

    def _toggle_theme(self) -> None:
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
        self.title_bar.set_palette(p)
        self.smart_bar.set_palette(p)
        self.list.set_palette(p)
        self.setWindowIcon(QIcon(str(APP_ICON)))
