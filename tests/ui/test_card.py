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


def test_card_status_label_fits_its_text(qtbot, qapp):
    from downloader.ui.theme import load_fonts, stylesheet
    load_fonts()
    before = qapp.styleSheet()
    qapp.setStyleSheet(stylesheet("dark"))  # as the app does: on the application, not the card
    try:
        card = _card(qtbot, _d(status=Status.QUEUED, downloaded=0))
        card.resize(900, 74)
        card.show()
        qtbot.waitExposed(card)
        card.update_from(_d(), speed=13_200_000, eta=150)  # status changes: queued -> downloading
        qtbot.wait(50)
        label = card.findChild(QLabel, "cardStatus")
        assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
    finally:
        qapp.setStyleSheet(before)


def test_card_just_started_says_starting(qtbot):
    d = _d(downloaded=0)
    card = _card(qtbot, d)
    card.update_from(d, speed=0.0, eta=None)
    assert _text(card, "cardStatus") == "Starting…"
    assert _text(card, "cardDetail") == "connecting · 8 connections"


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


# ---- checking, verified, clips, identical copies --------------------------------

import datetime

from PySide6.QtWidgets import QPushButton

from downloader.ui.card import status_texts
from downloader.ui.formatting import fmt_day

SHA = "sha256:" + "ab" * 32


def test_status_checking():
    status, _ = status_texts(_d(checking=True, downloaded=5_800_000_000), None, None)
    assert status == "Checking the file…"


def test_status_verified():
    _, detail = status_texts(_d(status=Status.COMPLETED, verified=True, checksum=SHA), None, None)
    assert detail.endswith("· ✓ SHA-256 verified")
    _, detail = status_texts(_d(status=Status.COMPLETED, verified=True, checksum="md5:" + "a" * 32), None, None)
    assert detail.endswith("· ✓ MD5 verified")


def test_status_paused_clip():
    d = _d(kind=Kind.VIDEO, status=Status.PAUSED, clip_start=750, clip_end=900, downloaded=100)
    assert "will start over · clips can't resume" in status_texts(d, None, None)[1]


def test_status_duplicate():
    _, detail = status_texts(_d(status=Status.COMPLETED, duplicate_of="setup.zip"), None, None)
    assert detail == "Same as setup.zip, which was already in this folder"


def test_card_meta_shows_clip(qtbot):
    card = _card(qtbot, _d(kind=Kind.VIDEO, quality="720p", clip_start=750, clip_end=900))
    assert "Clip 12:30–15:00 · 720p" in _text(card, "cardMeta")


def _links(card):
    return {b.text(): b for b in card.findChildren(QPushButton)}


def test_duplicate_links_visible_only_with_note(qtbot):
    card = _card(qtbot, _d(status=Status.COMPLETED))
    card.show()
    assert not _links(card)["Delete this copy"].isVisible()
    d = _d(status=Status.COMPLETED, duplicate_of="setup.zip")
    card.update_from(d)
    links = _links(card)
    assert links["Delete this copy"].isVisible() and links["Keep both"].isVisible()
    with qtbot.waitSignal(card.delete_copy_clicked) as sig:
        qtbot.mouseClick(links["Delete this copy"], Qt.LeftButton)
    assert sig.args == [7]
    with qtbot.waitSignal(card.keep_both_clicked):
        qtbot.mouseClick(links["Keep both"], Qt.LeftButton)


def test_card_with_duplicate_links_fits(qtbot):
    card = _card(qtbot, _d(status=Status.COMPLETED, duplicate_of="a-very-long-name-for-an-installer-setup.zip"))
    card.resize(900, 74)
    card.show()
    qtbot.waitExposed(card)
    links = _links(card)
    primary = _primary(card)
    right = max(b.geometry().translated(b.parentWidget().mapTo(card, b.pos()) - b.pos()).right()
                for b in links.values() if b.text() in ("Delete this copy", "Keep both"))
    assert right < primary.geometry().left()


def test_fmt_day():
    now = datetime.datetime(2026, 10, 27, 12).timestamp()
    assert fmt_day(datetime.datetime(2026, 10, 26, 9).timestamp(), now) == "26 Oct"
    assert fmt_day(datetime.datetime(2025, 10, 26, 9).timestamp(), now) == "26 Oct 2025"
