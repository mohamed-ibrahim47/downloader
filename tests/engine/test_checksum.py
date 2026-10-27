import hashlib
import socket
import time

import pytest

from downloader.engine.checksum import (Checksum, find_in_sums, hash_file, lookup_sums, parse_checksum)

H256 = hashlib.sha256(b"hello").hexdigest()


def test_parse_forms():
    assert parse_checksum(H256) == Checksum("sha256", H256)
    assert parse_checksum(f"  SHA256:{H256.upper()} ") == Checksum("sha256", H256)
    assert parse_checksum(f"{H256}  ubuntu.iso") == Checksum("sha256", H256)
    assert parse_checksum(f"{H256} *ubuntu.iso") == Checksum("sha256", H256)
    assert parse_checksum("a" * 32).algo == "md5"
    assert parse_checksum("a" * 40).algo == "sha1"
    assert parse_checksum("a" * 128).algo == "sha512"


@pytest.mark.parametrize("bad", ["", "xyz", "a" * 63, "g" * 64, f"md5:{H256}", "sha256:"])
def test_parse_rejects(bad):
    assert parse_checksum(bad) is None


def test_stored_form_and_label():
    c = Checksum("sha256", H256)
    assert str(c) == f"sha256:{H256}"
    assert Checksum.from_stored(str(c)) == c
    assert Checksum.from_stored(None) is None
    assert c.label == "SHA-256" and Checksum("sha1", "a" * 40).label == "SHA-1"


def test_find_in_sums_real_world_lines():
    text = f"# made by sha256sum\r\n{'b' * 64}  other.iso\r\n{H256.upper()} *./ubuntu.iso\r\n"
    assert find_in_sums(text, "ubuntu.iso", single_file=False) == Checksum("sha256", H256)
    assert find_in_sums(text, "missing.iso", single_file=False) is None


def test_single_file_sums_accepts_bare_hash():
    assert find_in_sums(f"{H256}\n", "ubuntu.iso", single_file=True) == Checksum("sha256", H256)
    assert find_in_sums(f"{H256}\n", "ubuntu.iso", single_file=False) is None


def test_single_file_sums_with_other_name_is_ignored():
    assert find_in_sums(f"{H256}  other.iso\n", "ubuntu.iso", single_file=True) is None


def test_hash_file_several_algos(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello")
    assert hash_file(p, {"sha256", "md5"}) == {"sha256": H256, "md5": hashlib.md5(b"hello").hexdigest()}


def test_hash_file_stops(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello" * 1000)
    assert hash_file(p, {"sha256"}, should_stop=lambda: True) is None


def test_lookup_finds_sha256sums_in_same_folder(http_server):
    http_server.add("/rel/SHA256SUMS", f"{H256}  app.zip\n".encode())
    url = http_server.url("/rel/app.zip?token=1")
    assert lookup_sums(url, "app.zip") == (Checksum("sha256", H256), "SHA256SUMS")
    tried = [p for (_, p, _) in http_server.requests]
    assert tried[:3] == ["/rel/app.zip.sha256", "/rel/app.zip.sha256sum", "/rel/SHA256SUMS"]


def test_lookup_prefers_per_file(http_server):
    http_server.add("/rel/app.zip.sha256", f"{H256}  app.zip\n".encode())
    http_server.add("/rel/SHA256SUMS", f"{'c' * 64}  app.zip\n".encode())
    assert lookup_sums(http_server.url("/rel/app.zip"), "app.zip") == (Checksum("sha256", H256), "app.zip.sha256")


def test_lookup_sha512sums(http_server):
    h = hashlib.sha512(b"x").hexdigest()
    http_server.add("/rel/SHA512SUMS", f"{h}  app.zip\n".encode())
    assert lookup_sums(http_server.url("/rel/app.zip"), "app.zip") == (Checksum("sha512", h), "SHA512SUMS")


def test_lookup_does_not_follow_redirects(http_server):
    http_server.add("/rel/app.zip.sha256", b"", redirect_to="/login/app.zip.sha256")
    http_server.add("/login/app.zip.sha256", f"{H256}\n".encode())
    assert lookup_sums(http_server.url("/rel/app.zip"), "app.zip") is None
    assert not any(p.startswith("/login") for (_, p, _) in http_server.requests)


def test_lookup_skips_oversized(http_server):
    big = f"{H256}  app.zip\n".encode() + b"#" * 2_000_000
    http_server.add("/rel/SHA256SUMS", big)
    assert lookup_sums(http_server.url("/rel/app.zip"), "app.zip") is None


def test_lookup_gives_up_on_slow_server(http_server):
    http_server.add("/rel/SHA256SUMS", f"{H256}  app.zip\n".encode() * 20000, delay_per_chunk=2)
    t = time.monotonic()
    assert lookup_sums(http_server.url("/rel/app.zip"), "app.zip", timeout=0.5) is None
    assert time.monotonic() - t < 3


def test_lookup_quiet_on_connection_error():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    t = time.monotonic()
    assert lookup_sums(f"http://127.0.0.1:{port}/rel/app.zip", "app.zip", timeout=1) is None
    assert time.monotonic() - t < 2.5  # an unreachable site isn't asked six times
