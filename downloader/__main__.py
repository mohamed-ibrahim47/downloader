"""Start the app: `python -m downloader` (or `--smoke` to open and close, for build checks)."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .engine.paths import data_home
    from .engine.settings import Settings
    from .engine.store import Store
    from .ui.controller import Controller
    from .ui.main_window import MainWindow
    from .ui.theme import load_fonts

    if sys.platform == "win32":
        import ctypes
        # Own taskbar group and icon, not python.exe's.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("downloader.app")

    app = QApplication.instance() or QApplication([sys.argv[0], *argv])
    app.setApplicationName("Downloader")
    load_fonts()
    store = Store(data_home() / "downloader.db")
    settings = Settings.load(store)
    controller = Controller(store, settings)
    window = MainWindow(controller, settings, store)
    # With a tray icon, closing the window while downloading keeps the app running.
    app.setQuitOnLastWindowClosed(window.tray is None)
    window.show()
    if "--smoke" in argv:
        QTimer.singleShot(1500, app.quit)
    code = app.exec()
    controller.shutdown()
    window.deleteLater()
    store.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
