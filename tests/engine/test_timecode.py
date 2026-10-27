import pytest

from downloader.engine.timecode import clip_label, clip_suffix, fmt_clock, parse_time, start_from_url


@pytest.mark.parametrize("text,seconds", [("750", 750), ("12:30", 750), ("1:02:03", 3723), ("12m30s", 750),
                                          ("1h2m3s", 3723), ("750s", 750), ("2m", 120), (" 12:30 ", 750),
                                          ("0", 0), ("90m", 5400)])
def test_parse_time(text, seconds):
    assert parse_time(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "12:75", "1:60:00", "-5", "12:30:", "1h75m", "12.5", "1s2m", "h"])
def test_parse_time_rejects(text):
    assert parse_time(text) is None


def test_fmt_clock():
    assert [fmt_clock(s) for s in (5, 750, 3723)] == ["0:05", "12:30", "1:02:03"]


@pytest.mark.parametrize("url,start", [
    ("https://www.youtube.com/watch?v=aqz-KE-bpKQ&t=750", 750),
    ("https://youtu.be/aqz-KE-bpKQ?t=12m30s", 750),
    ("https://www.youtube.com/watch?v=aqz-KE-bpKQ&start=1h2m3s", 3723),
    ("https://www.youtube.com/watch?v=aqz-KE-bpKQ#t=12:30", 750),
    ("https://www.youtube.com/watch?v=aqz-KE-bpKQ", None),
    ("https://www.youtube.com/watch?v=aqz-KE-bpKQ&t=junk", None)])
def test_start_from_url(url, start):
    assert start_from_url(url) == start


def test_clip_label_and_suffix():
    assert clip_label(750, 900) == "Clip 12:30–15:00"
    assert clip_suffix(750, 900) == " (12-30 to 15-00)"
    assert clip_suffix(3723, 3900) == " (1-02-03 to 1-05-00)"
