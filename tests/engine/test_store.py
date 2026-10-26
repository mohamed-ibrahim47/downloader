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
