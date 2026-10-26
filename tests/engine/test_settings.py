from downloader.engine.scheduler import Schedule
from downloader.engine.settings import Settings
from downloader.engine.store import Store


def test_defaults_match_spec():
    s = Settings()
    assert (s.max_active, s.connections, s.theme, s.when_done) == (3, 8, "dark", "nothing")
    assert s.watch_clipboard is True and s.number_playlist_files is True
    assert s.folder_video.endswith("Video") and s.folder_playlist.endswith("Playlists")


def test_settings_roundtrip(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    s = Settings(theme="light", max_active=5, speed_limit_on=True,
                 schedule=Schedule(enabled=True, start="03:30", stop="07:00", days=[0, 2]))
    s.save(store)
    assert Settings.load(store) == s


def test_settings_ignores_unknown_and_fills_missing(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    store.set_setting("settings", {"theme": "light", "bogus": 1, "schedule": {"enabled": True}})
    s = Settings.load(store)
    assert s.theme == "light"
    assert s.connections == 8
    assert s.schedule.enabled is True and s.schedule.start == "02:00"


def test_load_without_saved_settings(tmp_path):
    assert Settings.load(Store(tmp_path / "db.sqlite")) == Settings()
