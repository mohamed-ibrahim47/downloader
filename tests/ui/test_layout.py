"""The queue settings live on their own page (left menu), without a theme choice; the window
draws its own title bar in the theme's colours."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

from downloader.engine.settings import Settings
from downloader.engine.store import Store
from downloader.ui.controller import Controller
from downloader.ui.main_window import MainWindow
from downloader.ui.queue_panel import QueuePanel


class IdleJob:
    def __init__(self, d, p, f):
        self.download = d

    def start(self):
        pass

    def pause(self):
        pass


@pytest.fixture
def win(tmp_path, qtbot):
    store = Store(tmp_path / "db.sqlite")
    settings = Settings(folder_general=str(tmp_path))
    ctl = Controller(store, settings, job_factory=IdleJob)
    w = MainWindow(ctl, settings, store)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    yield w
    ctl.shutdown()
    store.close()


def test_queue_panel_has_no_theme_choice(qtbot):
    panel = QueuePanel(Settings())
    qtbot.addWidget(panel)
    labels = {b.text() for b in panel.findChildren(QPushButton)}
    assert "Dark" not in labels and "Light" not in labels


def test_queue_section_is_a_page_opened_from_the_menu(win):
    panel = win.findChild(QueuePanel)
    assert not panel.isVisible()  # not beside the downloads any more
    win._nav_buttons["queue"].click()
    assert win.pages.currentWidget() is win.queue_page
    assert panel.isVisible()
    win._nav_buttons["downloads"].click()
    assert not panel.isVisible()


def test_window_draws_its_own_title_bar(win):
    assert win.windowFlags() & Qt.FramelessWindowHint
    bar = win.title_bar
    assert bar.isVisible()
    assert bar.title.text() == "Downloader"


def test_title_bar_buttons(win, qtbot):
    bar = win.title_bar
    qtbot.mouseClick(bar.findChild(QToolButton, "maxButton"), Qt.LeftButton)
    assert win.isMaximized()
    qtbot.mouseDClick(bar.title, Qt.LeftButton)
    assert not win.isMaximized()
    assert bar.findChild(QToolButton, "closeButton").accessibleName() == "Close"


def test_title_bar_follows_theme(win, qtbot):
    qtbot.mouseClick(win.findChild(QToolButton, "themeButton"), Qt.LeftButton)
    assert "QFrame#titleBar" in QApplication.instance().styleSheet()
    assert win.title_bar.title.text() == "Downloader"


def test_edges_resize(win):
    from downloader.ui.title_bar import EdgeGrip
    grips = win.findChildren(EdgeGrip)
    assert len(grips) == 8  # 4 sides, 4 corners
    right = next(g for g in grips if g.edges == Qt.RightEdge)
    assert right.geometry().right() == win.width() - 1
    assert right.cursor().shape() == Qt.SizeHorCursor
    corner = next(g for g in grips if g.edges == (Qt.LeftEdge | Qt.TopEdge))
    assert corner.geometry().topLeft() == QPoint(0, 0)
    win.showMaximized()
    assert not right.isVisible()  # nothing to resize when maximised


def test_no_app_wide_event_filter(win):
    # An app-wide Python filter runs for every event of every widget and slows everything down.
    assert not hasattr(win, "eventFilter") or type(win).eventFilter is MainWindow.__mro__[1].eventFilter
