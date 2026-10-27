import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

CHUNK = 65536


@dataclass
class _Entry:
    data: bytes
    ranges: bool = True
    length: bool = True
    status: int = 200
    fail_times: int = 0
    fail_without_data: bool = False  # failures send nothing at all (no progress)
    disposition: str | None = None
    delay_per_chunk: float = 0.0
    max_conns: int = 0  # >0: answer 429 when more GETs than this are in flight
    active: int = 0
    redirect_to: str | None = None  # answer 302 to this path
    gzip: bool | str = False  # compress the body (as some servers do even for 206); "always": even unasked
    trickle: float = 0.0  # >0: send the body one byte at a time, this many seconds apart


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _entry(self):
        return self.server.owner.entries.get(self.path)

    def _send(self, body: bool):
        owner = self.server.owner
        entry = self._entry()
        rng = self.headers.get("Range")
        owner.requests.append((self.command, self.path, rng))
        if entry is None:
            self.send_response(404)
            self.end_headers()
            return
        if entry.redirect_to:
            self.send_response(302)
            self.send_header("Location", entry.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status_now = entry.status
        if body and entry.max_conns:
            with owner.lock:
                if entry.active >= entry.max_conns:
                    status_now = 429
                else:
                    entry.active += 1
        if status_now != 200:
            self.send_response(status_now)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        try:
            self._send_body(entry, rng, body)
        finally:
            if body and entry.max_conns:
                with owner.lock:
                    entry.active -= 1

    def _send_body(self, entry, rng, body):
        owner = self.server.owner

        data = entry.data
        start, end = 0, len(data) - 1
        status = 200
        if rng and entry.ranges:
            m = re.match(r"bytes=(\d+)-(\d*)", rng)
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else len(data) - 1
            status = 206
        part = data[start:end + 1]
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "") or entry.gzip == "always"
        if entry.gzip and accepts_gzip:
            import gzip as _gzip
            part = _gzip.compress(part)

        self.send_response(status)
        if entry.gzip and accepts_gzip:
            self.send_header("Content-Encoding", "gzip")
        if entry.ranges:
            self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        if entry.length:
            self.send_header("Content-Length", str(len(part)))
        if entry.disposition:
            self.send_header("Content-Disposition", entry.disposition)
        self.end_headers()
        if not body:
            return

        fail = False
        with owner.lock:
            if entry.fail_times > 0:
                entry.fail_times -= 1
                fail = True
        if fail:
            # Send half and hang up: the client sees a broken connection.
            if not entry.fail_without_data:
                self.wfile.write(part[: len(part) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        if entry.trickle:
            for i in range(len(part)):
                time.sleep(entry.trickle)
                try:
                    self.wfile.write(part[i:i + 1])
                    self.wfile.flush()
                except (ConnectionError, OSError):
                    return
            return
        for i in range(0, len(part), CHUNK):
            if entry.delay_per_chunk:
                time.sleep(entry.delay_per_chunk)
            try:
                self.wfile.write(part[i:i + CHUNK])
            except (ConnectionError, OSError):
                return

    def do_HEAD(self):
        self._send(body=False)

    def do_GET(self):
        self._send(body=True)


class TestServer:
    def __init__(self):
        self.entries: dict[str, _Entry] = {}
        self.requests: list[tuple[str, str, str | None]] = []
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.owner = self
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def add(self, path: str, data: bytes, **options) -> str:
        self.entries[path] = _Entry(data=data, **options)
        return self.url(path)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}{path}"

    def get_ranges(self, path: str) -> list[str | None]:
        return [r for (m, p, r) in self.requests if m == "GET" and p == path]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def http_server():
    server = TestServer()
    yield server
    server.close()
