import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _no_real_sums_lookup(monkeypatch):
    """The smart bar looks for SHA256SUMS next to every file link; UI tests never go online."""
    from downloader.ui import smart_bar
    monkeypatch.setattr(smart_bar, "lookup_sums", lambda url, name: None, raising=False)


@pytest.fixture(autouse=True)
def _no_real_youtube_search(monkeypatch):
    """A window built with the default Recommender searches through this, never YouTube."""
    from downloader.engine import youtube
    monkeypatch.setattr(youtube, "search", lambda query, limit=15: [])
