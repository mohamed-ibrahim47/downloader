from downloader.ui.formatting import fmt_bytes, fmt_eta, fmt_speed
from downloader.ui.icons import ICON_PATHS, icon
from downloader.ui.theme import PALETTES, stylesheet


def test_palettes_match_mockup():
    assert PALETTES["dark"]["bg"] == "#0E1015"
    assert PALETTES["dark"]["accent"] == "#6EA8FE"
    assert PALETTES["light"]["bg"] == "#F4F5F7"
    assert PALETTES["light"]["accent"] == "#2563EB"
    assert set(PALETTES["dark"]) == set(PALETTES["light"])


def test_stylesheet_has_palette_colours():
    assert "#0E1015" in stylesheet("dark")
    assert "#F4F5F7" in stylesheet("light")
    assert "#0E1015" not in stylesheet("light")


def test_icons_render(qapp):
    for name in ICON_PATHS:
        assert not icon(name, "#000000").isNull()


def test_fmt_bytes():
    assert fmt_bytes(None) == ""
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(48_200_000) == "48.2 MB"
    assert fmt_bytes(5_800_000_000) == "5.8 GB"


def test_fmt_speed():
    assert fmt_speed(8_400_000) == "8.4 MB/s"
    assert fmt_speed(0) == "0 B/s"


def test_fmt_duration():
    from downloader.ui.formatting import fmt_duration
    assert fmt_duration(None) == ""
    assert fmt_duration(596) == "9 min 56 s"
    assert fmt_duration(34_860) == "9 h 41 min"


def test_fmt_eta():
    assert fmt_eta(None) == ""
    assert fmt_eta(11) == "about 11 s left"
    assert fmt_eta(271) == "about 4 min left"
    assert fmt_eta(3900) == "about 1 h 5 min left"
