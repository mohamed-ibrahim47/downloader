"""Draws downloader/ui/assets/app.ico (and app.png): the download arrow on the accent square."""
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

from downloader.ui.icons import svg

ASSETS = Path(__file__).resolve().parents[1] / "downloader" / "ui" / "assets"


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#2563EB"))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)
    QSvgRenderer(svg("download", "#FFFFFF", 2.4).encode()).render(
        p, QRectF(size * 0.2, size * 0.2, size * 0.6, size * 0.6))
    p.end()
    return img


if __name__ == "__main__":
    app = QGuiApplication([])
    ASSETS.mkdir(parents=True, exist_ok=True)
    draw(256).save(str(ASSETS / "app.ico"))
    draw(256).save(str(ASSETS / "app.png"))
    print("wrote", ASSETS / "app.ico")
