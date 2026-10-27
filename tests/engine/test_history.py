import time

from downloader.engine.history import Existing, already_have, find_existing, same_link
from downloader.engine.models import Download, Kind, Status
from downloader.engine.youtube import PlaylistEntry

VID = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"


def make(tmp_path, name="a.zip", url="https://x.org/a.zip", kind=Kind.FILE, status=Status.COMPLETED,
         on_disk=True, **kw):
    d = Download(url=url, kind=kind, filename=name, folder=str(tmp_path), status=status,
                 finished_at=time.time() if status is Status.COMPLETED else None, **kw)
    if on_disk:
        (tmp_path / name).write_bytes(b"x")
    return d


def test_same_link():
    assert same_link("https://x.org/a.zip#frag", "https://x.org/a.zip")
    assert not same_link("https://x.org/a.zip?v=1", "https://x.org/a.zip?v=2")


def test_finished_file_same_link_ignoring_fragment(tmp_path):
    d = make(tmp_path)
    assert find_existing([d], "https://x.org/a.zip#x") == Existing(d, True)


def test_query_string_matters(tmp_path):
    d = make(tmp_path, url="https://x.org/a.zip?v=1")
    assert find_existing([d], "https://x.org/a.zip?v=2") is None


def test_completed_but_file_deleted_does_not_count(tmp_path):
    d = make(tmp_path, on_disk=False)
    assert find_existing([d], "https://x.org/a.zip") is None


def test_unfinished_in_list(tmp_path):
    for status in (Status.PAUSED, Status.QUEUED, Status.FAILED, Status.DOWNLOADING):
        d = make(tmp_path, status=status, on_disk=False)
        assert find_existing([d], "https://x.org/a.zip") == Existing(d, False)


def test_finished_preferred_over_in_list(tmp_path):
    queued = make(tmp_path, name="b.zip", status=Status.QUEUED, on_disk=False)
    done = make(tmp_path)
    assert find_existing([queued, done], "https://x.org/a.zip") == Existing(done, True)


def test_same_video_any_link_form(tmp_path):
    d = make(tmp_path, name="T.mp4", url=VID, kind=Kind.VIDEO, video_id="aqz-KE-bpKQ", quality="1080p")
    assert find_existing([d], "https://youtu.be/aqz-KE-bpKQ?t=5") == Existing(d, True)


def test_old_rows_without_video_id_match_by_url(tmp_path):
    d = make(tmp_path, name="T.mp4", url=VID + "&list=PL1", kind=Kind.PLAYLIST_ITEM)
    assert find_existing([d], "https://youtu.be/aqz-KE-bpKQ") == Existing(d, True)


def test_video_never_matches_a_file_row(tmp_path):
    d = make(tmp_path, url=VID)  # a FILE row whose url happens to be a watch link
    assert find_existing([d], VID) is None


def test_clip_matches_only_same_range(tmp_path):
    d = make(tmp_path, name="T.mp4", url=VID, kind=Kind.VIDEO, video_id="aqz-KE-bpKQ", clip_start=750, clip_end=900)
    assert find_existing([d], VID) is None
    assert find_existing([d], VID, clip=(750, 900)) == Existing(d, True)
    assert find_existing([d], VID, clip=(750, 901)) is None


def test_already_have_playlist(tmp_path):
    had = make(tmp_path, name="2.mp4", url="https://www.youtube.com/watch?v=BBBBBBBBBBB", kind=Kind.VIDEO,
               video_id="BBBBBBBBBBB")
    paused = make(tmp_path, name="3.mp4", url="https://www.youtube.com/watch?v=CCCCCCCCCCC", kind=Kind.VIDEO,
                  status=Status.PAUSED, video_id="CCCCCCCCCCC", on_disk=False)
    clip = make(tmp_path, name="1.mp4", url="https://www.youtube.com/watch?v=AAAAAAAAAAA", kind=Kind.VIDEO,
                video_id="AAAAAAAAAAA", clip_start=0, clip_end=10)
    entries = [PlaylistEntry(i, f"https://www.youtube.com/watch?v={c * 11}", f"V{i}", 60)
               for i, c in ((1, "A"), (2, "B"), (3, "C"))]
    assert already_have([had, paused, clip], entries) == {2}
