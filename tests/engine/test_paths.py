from downloader.engine.paths import data_home, safe_filename, unique_path


def test_data_home_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DOWNLOADER_HOME", str(tmp_path / "h"))
    assert data_home() == tmp_path / "h"
    assert (tmp_path / "h").is_dir()


def test_safe_filename_strips_illegal():
    assert safe_filename('a<b>:c"d/e\\f|g?h*.mp4') == "abcdefgh.mp4"


def test_safe_filename_trims_and_defaults():
    assert safe_filename("  name. ") == "name"
    assert safe_filename("???") == "download"


def test_safe_filename_keeps_unicode_and_caps_length():
    assert safe_filename("محاضرة 01.mp4") == "محاضرة 01.mp4"
    long = safe_filename("x" * 300 + ".zip")
    assert len(long) <= 200
    assert long.endswith(".zip")


def test_unique_path_adds_suffix(tmp_path):
    (tmp_path / "a.zip").write_bytes(b"1")
    (tmp_path / "a (2).zip").write_bytes(b"1")
    assert unique_path(tmp_path, "a.zip") == tmp_path / "a (3).zip"


def test_unique_path_free_name_unchanged(tmp_path):
    assert unique_path(tmp_path, "b.zip") == tmp_path / "b.zip"
