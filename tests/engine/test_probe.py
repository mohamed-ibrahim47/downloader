import pytest

from downloader.engine.errors import DownloadError
from downloader.engine.probe import probe


def test_probe_reads_size_and_ranges(http_server):
    url = http_server.add("/files/a.zip", b"x" * 1234)
    info = probe(url)
    assert info.size == 1234
    assert info.resumable is True
    assert info.filename == "a.zip"


def test_probe_without_ranges(http_server):
    url = http_server.add("/b.bin", b"x" * 10, ranges=False)
    assert probe(url).resumable is False


def test_probe_unknown_length(http_server):
    url = http_server.add("/c.bin", b"x" * 10, ranges=False, length=False)
    assert probe(url).size is None


def test_probe_filename_star(http_server):
    url = http_server.add(
        "/dl", b"x", disposition="attachment; filename*=UTF-8''%D9%85%D9%84%D9%81%3F.zip")
    assert probe(url).filename == "ملف.zip"


def test_probe_plain_disposition(http_server):
    url = http_server.add("/dl2", b"x", disposition='attachment; filename="report 2025.pdf"')
    assert probe(url).filename == "report 2025.pdf"


def test_probe_name_from_url_without_query(http_server):
    url = http_server.add("/path/My%20File.iso", b"x") + "?token=abc"
    http_server.entries["/path/My%20File.iso?token=abc"] = http_server.entries["/path/My%20File.iso"]
    assert probe(url).filename == "My File.iso"


def test_probe_403_raises_plain_message(http_server):
    url = http_server.add("/secret", b"x", status=403)
    with pytest.raises(DownloadError) as err:
        probe(url)
    assert str(err.value) == "server refused (403) · link may have expired"


def test_probe_404(http_server):
    with pytest.raises(DownloadError) as err:
        probe(http_server.url("/missing"))
    assert str(err.value) == "file not found on server (404)"


def test_probe_connection_refused():
    with pytest.raises(DownloadError) as err:
        probe("http://127.0.0.1:9/nothing", timeout=2)
    assert str(err.value) == "connection lost · check your internet"
