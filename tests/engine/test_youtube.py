import json
import os
from pathlib import Path

import pytest

from downloader.engine.youtube import (
    PlaylistEntry, PlaylistInfo, VideoInfo, choose_format, estimate_playlist_size,
    fetch_info, format_options, friendly_error,
)

FORMATS = json.loads((Path(__file__).parent / "fixtures" / "yt_formats.json").read_text())


def test_choose_720_exact():
    c = choose_format(FORMATS, "720p")
    assert c.height == 720
    assert "height<=720" in c.selector
    # H.264 first: AV1/VP9 in an .mp4 won't play in the stock Windows player.
    assert c.selector.startswith("bestvideo[height<=720][vcodec^=avc1]+bestaudio[ext=m4a]/")
    assert c.size == 44_800_000 + 3_400_000
    assert c.note is None
    assert c.ext == "mp4"


def test_choose_falls_back_lower():
    no_hd = [f for f in FORMATS if (f.get("height") or 0) < 1080]
    c = choose_format(no_hd, "1080p")
    assert c.height == 720
    assert c.note == "1080p not offered · got 720p"


def test_choose_uses_approx_size():
    assert choose_format(FORMATS, "1080p").size == 93_000_000 + 3_400_000


def test_choose_audio_is_m4a():
    c = choose_format(FORMATS, "Audio")
    assert c.ext == "m4a"
    assert c.height is None
    assert c.size == 3_400_000
    assert "bestaudio" in c.selector


def test_choose_lowest_when_all_above_target():
    only_hd = [f for f in FORMATS if f.get("height") in (None, 1080)]
    assert choose_format(only_hd, "480p").height == 1080


def test_format_options_order():
    opts = format_options(FORMATS)
    assert [o.id for o in opts] == ["v1440", "v1080", "v720", "v480", "v360", "v144", "m4a", "mp3"]
    assert opts[0].kind.startswith("WebM")
    assert opts[2].kind.startswith("MP4")
    assert opts[-1].audio_format == "mp3"
    assert opts[2].size == 44_800_000 + 3_400_000


def test_estimate_playlist_size():
    entries = [PlaylistEntry(1, "u1", "a", 3600), PlaylistEntry(2, "u2", "b", 3600)]
    assert estimate_playlist_size(entries, "720p") == 1_008_000_000


def test_estimate_playlist_size_unknown_durations():
    assert estimate_playlist_size([PlaylistEntry(1, "u", "a", None)], "720p") is None


@pytest.mark.parametrize("raw,expected", [
    ("ERROR: unable to download video data: HTTP Error 403: Forbidden",
     "server refused (403) · link may have expired"),
    ("ERROR: [youtube] abc: Video unavailable", "this video is unavailable or private"),
    ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access",
     "this video is unavailable or private"),
    ("ERROR: [youtube] abc: Sign in to confirm your age", "YouTube asks for a sign-in to watch this video"),
    ("ERROR: <urlopen error [Errno 11001] getaddrinfo failed>", "connection lost · check your internet"),
    ("ERROR: something odd happened\nmore", "YouTube error: something odd happened"),
])
def test_friendly_error(raw, expected):
    assert friendly_error(raw) == expected


def _job(tmp_path, filename="Song.m4a"):
    from downloader.engine.models import Download, Kind
    from downloader.engine.ratelimit import TokenBucket
    from downloader.engine.youtube import YouTubeJob
    d = Download(url="u", kind=Kind.VIDEO, filename=filename, folder=str(tmp_path), format_selector="x")
    return YouTubeJob(d, TokenBucket(None), lambda _d: None, lambda _d: None)


def test_final_path_comes_from_ytdlp_not_a_guess(tmp_path):
    # Review I4: a newer "Song.mp4" next to our "Song.m4a" must not be taken for ours.
    job = _job(tmp_path)
    opts = job._options()
    assert opts["updatetime"] is False
    (tmp_path / "Song.m4a").write_bytes(b"a" * 10)
    (tmp_path / "Song.mp4").write_bytes(b"v" * 99)  # newer, same stem
    opts["post_hooks"][0](str(tmp_path / "Song.m4a"))
    job._finish_names()
    assert job.download.filename == "Song.m4a"
    assert job.download.size == 10


def test_job_can_be_signalled_without_waiting(tmp_path):
    job = _job(tmp_path)
    job.signal_pause()
    assert job._pause.is_set()


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("RUN_NETWORK") != "1", reason="set RUN_NETWORK=1")
def test_fetch_info_real_video():
    info = fetch_info("https://www.youtube.com/watch?v=aqz-KE-bpKQ&list=PLxxx")
    assert isinstance(info, VideoInfo)
    assert "Big Buck Bunny" in info.title
    assert choose_format(info.formats, "480p").height == 480


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("RUN_NETWORK") != "1", reason="set RUN_NETWORK=1")
def test_fetch_info_real_playlist():
    info = fetch_info("https://www.youtube.com/playlist?list=PLRqwX-V7Uu6ZiZxtDDRCi6uhfTH4FilpH")
    assert isinstance(info, PlaylistInfo)
    assert len(info.entries) > 3
    assert info.entries[0].index == 1


# ---- clips ---------------------------------------------------------------------

def _clip_job(tmp_path, clip=(10, 20)):
    from downloader.engine.models import Download, Kind
    from downloader.engine.ratelimit import TokenBucket
    from downloader.engine.youtube import YouTubeJob
    start, end = clip if clip else (None, None)
    d = Download(url="https://www.youtube.com/watch?v=x", kind=Kind.VIDEO, filename="T (0-10 to 0-20).mp4",
                 folder=str(tmp_path), format_selector="best", clip_start=start, clip_end=end)
    return YouTubeJob(d, TokenBucket(None), lambda _d: None, lambda _d: None, backoff=())


def test_clip_size():
    from downloader.engine.youtube import clip_size
    assert clip_size(48_200_000, 596, 0, 298) == 24_100_000
    assert clip_size(None, 596, 0, 10) is None
    assert clip_size(48_200_000, None, 0, 10) is None


def test_clip_options(tmp_path):
    opts = _clip_job(tmp_path)._options()
    assert opts["force_keyframes_at_cuts"] is True and opts["continuedl"] is False
    assert list(opts["download_ranges"]({}, None)) == [{"start_time": 10, "end_time": 20}]


def test_full_video_options_unchanged(tmp_path):
    opts = _clip_job(tmp_path, clip=None)._options()
    assert "download_ranges" not in opts and "force_keyframes_at_cuts" not in opts
    assert opts["continuedl"] is True


def test_clip_restarts_from_scratch(tmp_path, monkeypatch):
    import yt_dlp
    leftover = tmp_path / "T (0-10 to 0-20).f137.mp4.part"
    leftover.write_bytes(b"old")
    seen = []

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            seen.append(leftover.exists())

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    job = _clip_job(tmp_path)
    job.download.downloaded = 500
    job.run()
    assert seen == [False]


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("RUN_NETWORK") != "1", reason="set RUN_NETWORK=1")
def test_real_clip_downloads(tmp_path):
    from downloader.engine.models import Download, Kind, Status
    from downloader.engine.ratelimit import TokenBucket
    from downloader.engine.youtube import YouTubeJob, selector_for
    d = Download(url="https://www.youtube.com/watch?v=aqz-KE-bpKQ", kind=Kind.VIDEO,
                 filename="Bunny (0-10 to 0-20).mp4", folder=str(tmp_path), format_selector=selector_for("480p"),
                 clip_start=10, clip_end=20)
    YouTubeJob(d, TokenBucket(None), lambda _d: None, lambda _d: None).run()
    assert d.status is Status.COMPLETED, d.error
    assert d.target_path.exists() and d.target_path.stat().st_size < 3_000_000
