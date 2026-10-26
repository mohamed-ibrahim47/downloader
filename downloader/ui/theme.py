"""The two colour themes (copied from mockup board B) and the stylesheet built from them."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontDatabase

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0E1015", "rail": "#0A0C10", "railBorder": "#1E222B", "surface": "#171A21", "surface2": "#1C2029",
        "aside": "#12151B", "border": "#2A303C", "cardBorder": "#232833", "track": "#262B35", "text": "#E8EAED",
        "muted": "#A0A8B8", "accent": "#6EA8FE", "onAccent": "#0A0C10", "navActive": "#1E2430",
        "chipOn": "#E8EAED", "chipOnText": "#0A0C10", "badgeBg": "#1E2A40", "badgeText": "#A9CBFF",
        "pillOn": "#2A3242", "paused": "#F5A524", "error": "#FF7A7A", "done": "#4ADE80", "queued": "#A0A8B8",
    },
    "light": {
        "bg": "#F4F5F7", "rail": "#FFFFFF", "railBorder": "#E3E6EB", "surface": "#FFFFFF", "surface2": "#F7F8FA",
        "aside": "#FAFBFC", "border": "#D5DAE1", "cardBorder": "#E6E9EE", "track": "#E6E9EE", "text": "#15181E",
        "muted": "#566070", "accent": "#2563EB", "onAccent": "#FFFFFF", "navActive": "#E7EEFD",
        "chipOn": "#15181E", "chipOnText": "#FFFFFF", "badgeBg": "#E7EEFD", "badgeText": "#1D4ED8",
        "pillOn": "#FFFFFF", "paused": "#B45309", "error": "#C2362B", "done": "#15803D", "queued": "#566070",
    },
}

STATUS_COLOR = {"downloading": "accent", "queued": "queued", "paused": "paused", "failed": "error",
                "completed": "done"}

FONTS = Path(__file__).parent / "assets" / "fonts"
SANS = '"Inter Variable", "Inter", "Segoe UI", sans-serif'
MONO = '"Cascadia Mono", "Consolas", monospace'


def load_fonts() -> None:
    for f in FONTS.glob("*.ttf"):
        QFontDatabase.addApplicationFont(str(f))


def status_color(palette: dict[str, str], status: str) -> str:
    return palette[STATUS_COLOR[status]]


def stylesheet(theme: str) -> str:
    c = PALETTES[theme]
    status_rules = "\n".join(
        f'DownloadCard[status="{s}"] QLabel#cardStatus {{ color: {c[key]}; }}\n'
        f'DownloadCard[status="{s}"] QProgressBar#cardBar::chunk {{ background: {c[key]}; border-radius: 3px; }}'
        for s, key in STATUS_COLOR.items())
    return f"""
* {{ font-family: {SANS}; font-size: 13px; color: {c['text']}; }}
QMainWindow, QWidget#root, QWidget#page {{ background: {c['bg']}; }}
QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}

QFrame#rail {{ background: {c['rail']}; border-right: 1px solid {c['railBorder']}; }}
QLabel#logo {{ background: {c['accent']}; border-radius: 12px; }}
QToolButton#railButton {{ background: transparent; border: none; border-radius: 12px; }}
QToolButton#railButton:hover {{ background: {c['navActive']}; }}
QToolButton#railButton:checked {{ background: {c['navActive']}; }}
QToolButton#themeButton {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 12px; }}

QLabel#h1 {{ font-size: 22px; font-weight: 700; }}
QLabel#h2 {{ font-size: 14px; font-weight: 600; }}
QLabel#muted, QLabel#panelLabel, QLabel#hint {{ color: {c['muted']}; font-size: 12px; }}
QLabel#speedTotal {{ color: {c['muted']}; font-size: 12px; }}

QPushButton#chip {{ height: 28px; padding: 0 12px; border-radius: 15px; border: 1px solid {c['border']};
    background: transparent; color: {c['muted']}; font-size: 12px; font-weight: 500; }}
QPushButton#chip:checked {{ background: {c['chipOn']}; color: {c['chipOnText']}; border-color: {c['chipOn']}; }}

QFrame#smartBar {{ background: {c['surface']}; border: 2px solid {c['accent']}; border-radius: 14px; }}
QLineEdit#smartInput {{ background: transparent; border: none; font-family: {MONO}; font-size: 13px;
    selection-background-color: {c['accent']}; selection-color: {c['onAccent']}; }}
QLabel#badge {{ background: {c['badgeBg']}; color: {c['badgeText']}; border-radius: 8px; padding: 4px 10px;
    font-size: 12px; font-weight: 600; }}
QFrame#smartOptions {{ background: {c['surface2']}; border: none; border-top: 1px solid {c['cardBorder']};
    border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; }}
QLabel#thumb {{ background: {c['surface']}; border: 1px solid {c['cardBorder']}; border-radius: 8px; }}
QLabel#optTitle {{ font-size: 14px; font-weight: 600; }}
QLabel#optMeta, QLabel#sizeNote {{ color: {c['muted']}; font-size: 12px; }}
QLabel#sizeLabel {{ font-weight: 600; }}
QFrame#qualityGroup {{ background: {c['bg']}; border: 1px solid {c['cardBorder']}; border-radius: 10px; }}
QPushButton#qualityButton {{ height: 28px; padding: 0 11px; border: none; border-radius: 8px;
    background: transparent; color: {c['muted']}; font-size: 12px; }}
QPushButton#qualityButton:checked {{ background: {c['pillOn']}; color: {c['text']}; font-weight: 600; }}
QPushButton#primaryButton {{ min-height: 40px; padding: 0 18px; border: none; border-radius: 10px;
    background: {c['accent']}; color: {c['onAccent']}; font-weight: 600; }}
QPushButton#primaryButton:disabled {{ background: {c['border']}; color: {c['muted']}; }}
QPushButton#linkButton {{ border: none; background: transparent; color: {c['accent']}; font-size: 12px; padding: 0; }}
QPushButton#secondaryButton {{ min-height: 34px; padding: 0 14px; border: 1px solid {c['border']};
    border-radius: 10px; background: {c['surface']}; }}

DownloadCard {{ background: {c['surface']}; border: 1px solid {c['cardBorder']}; border-radius: 12px; }}
QLabel#cardIconBox {{ background: {c['surface2']}; border-radius: 10px; }}
QLabel#cardName {{ font-size: 14px; font-weight: 600; }}
QLabel#cardMeta, QLabel#cardDetail {{ color: {c['muted']}; font-size: 12px; }}
QLabel#cardStatus {{ font-size: 12px; font-weight: 600; }}
QProgressBar#cardBar {{ background: {c['track']}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }}
{status_rules}
QToolButton#cardPrimary, QToolButton#cardRemove {{ background: transparent; border: 1px solid {c['border']};
    border-radius: 10px; }}
QToolButton#cardPrimary:hover, QToolButton#cardRemove:hover {{ background: {c['surface2']}; }}
QPushButton#completedRow {{ min-height: 44px; padding: 0 14px; text-align: left; border-radius: 12px;
    border: 1px dashed {c['border']}; background: transparent; color: {c['muted']}; }}

QFrame#queuePanel {{ background: {c['aside']}; border-left: 1px solid {c['railBorder']}; }}
QPushButton#stepButton {{ min-width: 36px; min-height: 36px; border-radius: 10px; border: 1px solid {c['border']};
    background: {c['surface']}; font-size: 16px; }}
QLabel#stepValue {{ font-size: 16px; font-weight: 600; }}
QPushButton#segButton {{ min-height: 34px; border-radius: 8px; border: 1px solid {c['border']};
    background: transparent; font-size: 12px; }}
QPushButton#segButton:checked {{ background: {c['navActive']}; border-color: {c['accent']}; }}
QComboBox, QSpinBox, QTimeEdit, QLineEdit#field {{ min-height: 34px; border-radius: 10px; border: 1px solid {c['border']};
    background: {c['surface']}; padding: 0 10px; }}
QComboBox QAbstractItemView {{ background: {c['surface']}; border: 1px solid {c['border']};
    selection-background-color: {c['navActive']}; }}
QCheckBox {{ spacing: 10px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid {c['border']};
    background: {c['surface']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}
QProgressBar#diskBar {{ background: {c['track']}; border: none; border-radius: 2px; max-height: 4px; }}
QProgressBar#diskBar::chunk {{ background: {c['muted']}; border-radius: 2px; }}

QDialog {{ background: {c['bg']}; }}
QListWidget {{ background: {c['surface']}; border: 1px solid {c['cardBorder']}; border-radius: 10px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
