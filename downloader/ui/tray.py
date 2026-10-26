"""The tray icon: keeps downloads going with the window closed, and shows finished/failed messages."""
from __future__ import annotations

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from .theme import APP_ICON


class Tray(QSystemTrayIcon):
    def __init__(self, window, controller, accent: str):
        super().__init__(QIcon(str(APP_ICON)), window)
        self.setToolTip("Downloader")
        menu = QMenu(window)
        menu.addAction("Show", window.bring_to_front)
        menu.addSeparator()
        menu.addAction("Pause all", controller.stop_queue)
        menu.addAction("Resume all", controller.start_queue)
        menu.addSeparator()
        menu.addAction("Quit", window.quit_app)
        self.setContextMenu(menu)
        self.activated.connect(lambda reason: window.bring_to_front()
                               if reason == QSystemTrayIcon.Trigger else None)

    def notify(self, title: str, text: str) -> None:
        self.showMessage(title, text, QSystemTrayIcon.Information, 5000)
