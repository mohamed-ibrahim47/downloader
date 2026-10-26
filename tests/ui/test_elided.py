from downloader.ui.elided import ElidedLabel


def test_long_text_elides_and_keeps_full_text(qtbot):
    label = ElidedLabel(objectName="cardName")
    qtbot.addWidget(label)
    full = "Big Buck Bunny 60fps 4K - Official Blender Foundation Short Film " * 3
    label.setText(full)
    label.resize(200, 20)
    label.show()
    qtbot.waitExposed(label)
    assert label.text() == full
    assert label.shown_text().endswith("…")
    assert len(label.shown_text()) < len(full)
    assert label.minimumSizeHint().width() <= 40


def test_short_text_unchanged(qtbot):
    label = ElidedLabel()
    qtbot.addWidget(label)
    label.setText("a.zip")
    label.resize(300, 20)
    label.show()
    assert label.shown_text() == "a.zip"


def test_markup_stays_plain(qtbot):
    label = ElidedLabel()
    qtbot.addWidget(label)
    label.setText("<b>x</b>")
    assert "<b>x</b>" in label.shown_text()
