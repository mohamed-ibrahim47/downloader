import pytest

from downloader.engine.linkkind import LinkKind, classify


@pytest.mark.parametrize("url,kind", [
    ("https://example.com/a.zip", LinkKind.FILE),
    ("  https://example.com/a.zip?x=1  ", LinkKind.FILE),
    ("http://example.com/", LinkKind.FILE),
    ("https://www.youtube.com/watch?v=aqz-KE-bpKQ", LinkKind.VIDEO),
    ("https://youtube.com/watch?v=aqz-KE-bpKQ", LinkKind.VIDEO),
    ("https://youtu.be/aqz-KE-bpKQ", LinkKind.VIDEO),
    ("https://www.youtube.com/shorts/abc123", LinkKind.VIDEO),
    ("https://m.youtube.com/watch?v=x&list=PL1", LinkKind.VIDEO),
    ("https://music.youtube.com/watch?v=x", LinkKind.VIDEO),
    ("https://www.youtube.com/playlist?list=PL1", LinkKind.PLAYLIST),
    ("https://www.youtube.com/@channel", LinkKind.INVALID),
    ("https://www.youtube.com/watch", LinkKind.INVALID),
    ("https://youtu.be/", LinkKind.INVALID),
    ("hello world", LinkKind.INVALID),
    ("ftp://example.com/a", LinkKind.INVALID),
    ("", LinkKind.INVALID),
])
def test_classify(url, kind):
    assert classify(url) is kind


@pytest.mark.parametrize("url", ["https://www.youtube.com/watch?v=aqz-KE-bpKQ", "https://youtu.be/aqz-KE-bpKQ?si=x",
                                 "https://www.youtube.com/shorts/aqz-KE-bpKQ",
                                 "https://m.youtube.com/watch?v=aqz-KE-bpKQ&list=PL1&t=5",
                                 "https://music.youtube.com/watch?v=aqz-KE-bpKQ"])
def test_video_id(url):
    from downloader.engine.linkkind import video_id
    assert video_id(url) == "aqz-KE-bpKQ"


def test_video_id_none():
    from downloader.engine.linkkind import video_id
    assert video_id("https://example.com/a.zip") is None
    assert video_id("https://www.youtube.com/playlist?list=PL1") is None
    assert video_id("not a link") is None
