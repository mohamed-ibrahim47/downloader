import time

from downloader.engine.models import Download, Kind, Segment, Status
from downloader.engine.store import Store


def _download(**kw):
    base = dict(url="https://example.com/a.zip", kind=Kind.FILE, filename="a.zip", folder="D:/x")
    base.update(kw)
    return Download(**base)


def test_add_get_roundtrip(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    d = _download(
        size=1000, status=Status.PAUSED, resumable=True, connections=4, error=None, note="n",
        downloaded=300, quality="720p", format_selector="best", audio_format="mp3", subtitles=True,
        title="T", playlist_title="P", playlist_index=3,
        segments=[Segment(0, 0, 499, 200), Segment(1, 500, 999, 100)],
    )
    new_id = store.add(d)
    assert d.id == new_id
    got = store.get(new_id)
    assert got == d


def test_update_changes_row(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    d = _download()
    store.add(d)
    d.status = Status.FAILED
    d.error = "disk is full"
    store.update(d)
    assert store.get(d.id).error == "disk is full"
    assert store.get(d.id).status is Status.FAILED


def test_all_newest_first(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    old = _download(added_at=100.0)
    new = _download(added_at=200.0)
    store.add(old)
    store.add(new)
    assert [d.id for d in store.all()] == [new.id, old.id]


def test_delete(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    d = _download()
    store.add(d)
    store.delete(d.id)
    assert store.get(d.id) is None


def test_save_segments(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    d = _download()
    store.add(d)
    store.save_segments(d.id, [Segment(0, 0, 9, 9)])
    assert store.get(d.id).segments == [Segment(0, 0, 9, 9)]


def test_save_segments_updates_downloaded(tmp_path):
    # Review I1: after a crash the card must show the saved progress.
    store = Store(tmp_path / "db.sqlite")
    d = _download(size=100)
    store.add(d)
    store.save_segments(d.id, [Segment(0, 0, 49, 20), Segment(1, 50, 99, 15)])
    assert store.get(d.id).downloaded == 35


def test_add_many_is_one_fast_transaction(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    ds = [_download(url=f"https://e.com/{i}") for i in range(250)]
    t0 = time.monotonic()
    ids = store.add_many(ds)
    assert time.monotonic() - t0 < 2
    assert ids == [d.id for d in ds] and len(set(ids)) == 250
    assert len(store.all()) == 250


def test_settings_roundtrip(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    assert store.get_setting("k", "dflt") == "dflt"
    store.set_setting("k", {"a": [1, 2]})
    assert store.get_setting("k") == {"a": [1, 2]}


def test_open_marks_downloading_as_paused(tmp_path):
    path = tmp_path / "db.sqlite"
    store = Store(path)
    d = _download(status=Status.DOWNLOADING, downloaded=1234)
    store.add(d)
    store.close()
    reopened = Store(path)
    got = reopened.get(d.id)
    assert got.status is Status.PAUSED
    assert got.downloaded == 1234


def test_download_paths_and_host():
    d = _download(folder="D:/x", filename="a.zip", url="https://cdn.example.com/a.zip?x=1")
    assert d.part_path.name == "a.zip.part"
    assert d.target_path.name == "a.zip"
    assert d.host == "cdn.example.com"


OLD_COLUMNS = ["url", "kind", "filename", "folder", "size", "status", "resumable", "connections", "added_at",
               "finished_at", "error", "note", "downloaded", "quality", "format_selector", "audio_format",
               "subtitles", "title", "playlist_title", "playlist_index"]


def test_old_database_upgrades_in_place(tmp_path):
    import sqlite3

    db = sqlite3.connect(tmp_path / "db.sqlite")
    db.execute(f"CREATE TABLE downloads (id INTEGER PRIMARY KEY, {', '.join(OLD_COLUMNS)}, segments TEXT)")
    db.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT INTO downloads (url, kind, filename, folder, status, resumable, connections, added_at, "
               "downloaded, subtitles, segments) VALUES ('https://x/a.zip','file','a.zip','D:/dl','completed',1,8,"
               "1.0,5,0,'[]')")
    db.commit()
    db.close()
    store = Store(tmp_path / "db.sqlite")
    d = store.get(1)
    assert (d.filename, d.status, d.downloaded, d.resumable) == ("a.zip", Status.COMPLETED, 5, True)
    assert (d.clip_start, d.video_id, d.checksum, d.verified, d.checking, d.duplicate_of) == \
        (None, None, None, False, False, None)
    d.verified = True
    store.update(d)
    assert store.get(1).verified is True
    assert len(store.all()) == 1


def test_new_fields_roundtrip(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    d = _download(clip_start=750, clip_end=900, video_id="aqz-KE-bpKQ", checksum="sha256:" + "ab" * 32,
                  checksum_source="SHA256SUMS", verified=True, duplicate_of="setup.zip")
    store.add(d)
    assert store.get(d.id) == d


def test_open_clears_checking(tmp_path):
    path = tmp_path / "db.sqlite"
    store = Store(path)
    d = _download(status=Status.DOWNLOADING, checking=True)
    store.add(d)
    store.close()
    got = Store(path).get(d.id)
    assert got.status is Status.PAUSED and got.checking is False


def test_clip_property():
    assert _download(clip_start=750, clip_end=900).clip == (750, 900)
    assert _download().clip is None
