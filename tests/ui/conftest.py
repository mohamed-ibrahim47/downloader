import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _no_real_sums_lookup(monkeypatch):
    """The smart bar looks for SHA256SUMS next to every file link; UI tests never go online."""
    from downloader.ui import smart_bar
    monkeypatch.setattr(smart_bar, "lookup_sums", lambda url, name: None, raising=False)
