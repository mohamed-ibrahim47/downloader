from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QToolButton

from downloader.engine.models import Download, Kind, Status
from downloader.ui.card import DownloadCard
from downloader.ui.theme import PALETTES


def _d(**kw):
    base = dict(id=7, url="https://releases.ubuntu.com/a.iso", kind=Kind.FILE,
                filename="ubuntu-24.04.2-desktop-amd64.iso", folder="D:/dl", size=5_800_000_000,
                status=Status.DOWNLOADING, resumable=True, downloaded=3_596_000_000, connections=8)
    base.update(kw)
    return Download(**base)


def _card(qtbot, d):
    card = DownloadCard(d, PALETTES["dark"])
    qtbot.addWidget(card)
    return card


def _text(card, name):
    return card.findChild(QLabel, name).text()


def _primary(card):
    return card.findChild(QToolButton, "cardPrimary")


def test_card_downloading_text(qtbot):
    card = _card(qtbot, _d())
    card.update_from(_d(), speed=8_400_000, eta=271)
    assert _text(card, "cardName") == "ubuntu-24.04.2-desktop-amd64.iso"
    assert _text(card, "cardMeta") == "5.8 GB · releases.ubuntu.com"
    assert _text(card, "cardStatus") == "62% · 8.4 MB/s"
    assert _text(card, "cardDetail") == "3.6 GB done · about 4 min left · 8 connections"
    assert card.findChild(QProgressBar, "cardBar").value() == 62
    assert _primary(card).accessibleName() == "Pause"
    assert card.property("status") == "downloading"


def test_card_failed_shows_error_and_retry(qtbot):
    d = _d(status=Status.FAILED, error="server refused (403) · link may have expired")
    card = _card(qtbot, d)
    assert _text(card, "cardStatus") == "Failed"
    assert _text(card, "cardDetail") == "server refused (403) · link may have expired"
    assert _primary(card).accessibleName() == "Retry"


def test_card_paused_and_queued(qtbot):
    card = _card(qtbot, _d(status=Status.PAUSED, downloaded=1_972_000_000))
    assert _text(card, "cardStatus") == "Paused at 34%"
    assert _text(card, "cardDetail") == "resumes where it stopped"
    assert _primary(card).accessibleName() == "Resume"
    card.update_from(_d(status=Status.QUEUED, downloaded=0, kind=Kind.PLAYLIST_ITEM,
                        playlist_index=7, playlist_title="Algorithms course"))
    assert _text(card, "cardStatus") == "Waiting"
    assert _text(card, "cardDetail") == "video 7 in “Algorithms course”"
    assert _primary(card).accessibleName() == "Start now"


def test_card_paused_not_resumable_warns(qtbot):
    card = _card(qtbot, _d(status=Status.PAUSED, resumable=False))
    assert _text(card, "cardDetail") == "will start over · this server can't resume"


def test_card_unknown_size_indeterminate(qtbot):
    d = _d(size=None, downloaded=1_500_000)
    card = _card(qtbot, d)
    card.update_from(d, speed=1_000_000, eta=None)
    assert card.findChild(QProgressBar, "cardBar").maximum() == 0
    assert _text(card, "cardStatus") == "1.5 MB · 1.0 MB/s"


def test_card_note_shown(qtbot):
    d = _d(kind=Kind.VIDEO, quality="720p", note="1080p not offered · got 720p", status=Status.QUEUED)
    card = _card(qtbot, d)
    assert "1080p not offered · got 720p" in _text(card, "cardDetail")
    assert _text(card, "cardMeta") == "5.8 GB · 720p · releases.ubuntu.com"


def test_card_completed(qtbot):
    card = _card(qtbot, _d(status=Status.COMPLETED, downloaded=5_800_000_000))
    assert _text(card, "cardStatus") == "Completed"
    assert _primary(card).accessibleName() == "Open folder"


def test_card_shows_markup_as_plain_text(qtbot):
    d = _d(filename="<b>bold</b>.zip", url="https://x.com/a", quality="<i>q</i>",
           status=Status.FAILED, error="<img src='file:///C:/x.png'>")
    card = _card(qtbot, d)
    for name in ("cardName", "cardMeta", "cardStatus", "cardDetail"):
        assert card.findChild(QLabel, name).textFormat() == Qt.PlainText
    assert "&lt;b&gt;" in card.findChild(QLabel, "cardName").toolTip()
    assert "&lt;img" in card.findChild(QLabel, "cardDetail").toolTip()


def test_card_primary_emits_id(qtbot):
    card = _card(qtbot, _d())
    with qtbot.waitSignal(card.primary_clicked) as sig:
        qtbot.mouseClick(_primary(card), Qt.LeftButton)
    assert sig.args == [7]
    remove = card.findChild(QToolButton, "cardRemove")
    with qtbot.waitSignal(card.remove_clicked) as sig:
        qtbot.mouseClick(remove, Qt.LeftButton)
    assert sig.args == [7]
