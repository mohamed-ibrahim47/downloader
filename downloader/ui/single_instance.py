"""Only one copy of the app runs: a second launch asks the first to show its window, then quits.

Two copies would both write to the same database and the same .part files. Which copy is first
is decided by a lock file in the user's own data folder, which other users can't create; the
local socket is only the doorbell that brings the running window forward.
"""
from __future__ import annotations

import getpass
import hashlib

from PySide6.QtCore import QLockFile, QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

_KNOCK = b"downloader:show"
_ANSWER = b"downloader:ok"


def default_name(data_folder: str = "") -> str:
    tag = hashlib.sha1(data_folder.lower().encode()).hexdigest()[:10]
    return f"downloader-{getpass.getuser()}-{tag}"


class SingleInstance(QObject):
    activated = Signal()  # another launch asked us to come to the front

    def __init__(self, name: str, lock_path: str, parent=None):
        super().__init__(parent)
        self._name = name
        self._lock = QLockFile(lock_path)
        self._lock.setStaleLockTime(0)  # a crashed copy's lock is detected by its dead process id
        self._server: QLocalServer | None = None

    def try_lock(self) -> bool:
        """True if this is the only copy. False if another one runs (it has been asked to show)."""
        if not self._lock.tryLock(100):
            self._knock()
            return False
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.UserAccessOption)
        server.newConnection.connect(self._on_connection)
        if not server.listen(self._name):
            QLocalServer.removeServer(self._name)  # leftover from a crash; if someone else holds
            server.listen(self._name)             # it, we just can't be brought forward
        self._server = server
        return True

    def release(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self._lock.unlock()

    def _knock(self) -> None:
        socket = QLocalSocket()
        socket.connectToServer(self._name)
        if socket.waitForConnected(500):
            socket.write(_KNOCK)
            socket.flush()
            socket.waitForBytesWritten(500)
            # Hang up only once the running copy has read it (hanging up early loses the data).
            socket.waitForReadyRead(2000)
            socket.disconnectFromServer()

    def _on_connection(self) -> None:
        while self._server is not None and self._server.hasPendingConnections():
            conn = self._server.nextPendingConnection()
            conn.setProperty("heard", b"")
            conn.readyRead.connect(lambda c=conn: self._on_knock(c))
            conn.disconnected.connect(lambda c=conn: (self._on_knock(c), c.deleteLater()))
            self._on_knock(conn)  # the knock may have arrived before readyRead was connected

    def _on_knock(self, conn: QLocalSocket) -> None:
        heard = bytes(conn.property("heard") or b"") + bytes(conn.readAll())
        conn.setProperty("heard", heard)
        if heard.startswith(_KNOCK) and not conn.property("answered"):
            conn.setProperty("answered", True)
            conn.write(_ANSWER)
            conn.flush()
            self.activated.emit()
