"""Stroke icons drawn from SVG path data (the same paths as the mockup), tinted per theme."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICON_PATHS = {
    "download": "M12 4v12M6 10l6 6 6-6M4 20h16",
    "list": "M4 6h16M4 12h16M4 18h16",
    "video": "M3 6h13v12H3zM16 10l5-3v10l-5-3",
    "playlist": "M4 6h12M4 12h12M4 18h8M17 15l4 3-4 3z",
    "file": "M6 3h8l4 4v14H6zM14 3v4h4",
    "folder": "M3 6h6l2 2h10v11H3z",
    "clock": "M3 12a9 9 0 1 0 18 0a9 9 0 1 0-18 0M12 7v5l3 2",
    "sliders": "M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1M15 4v4M9 10v4M17 16v4",
    "link": "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    "archive": "M4 4h16v4H4zM5 8v12h14V8M10 12h4",
    "music": "M9 18V5l11-2v13M9 18a3 3 0 1 1-6 0a3 3 0 1 1 6 0M20 16a3 3 0 1 1-6 0a3 3 0 1 1 6 0",
    "play": "M7 5l12 7-12 7z",
    "pause": "M8 5v14M16 5v14",
    "retry": "M4 12a8 8 0 1 0 2.3-5.7M4 4v4h4",
    "check": "M5 12l5 5 9-10",
    "x": "M6 6l12 12M18 6L6 18",
    "sun": "M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4"
           "M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
    "moon": "M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z",
}


def svg(name: str, color: str, stroke: float = 1.8) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="{ICON_PATHS[name]}"/></svg>')


@lru_cache(maxsize=512)
def pixmap(name: str, color: str, size: int = 20, stroke: float = 1.8, ratio: float = 2.0) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(svg(name, color, stroke).encode()))
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(ratio)
    return pm


def icon(name: str, color: str, size: int = 20, stroke: float = 1.8) -> QIcon:
    return QIcon(pixmap(name, color, size, stroke))
