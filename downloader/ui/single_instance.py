"""Only one copy of the app runs: a second launch asks the first to show its window, then quits.

Two copies would both write to the same database and the same .part files.
"""
from __future__ import annotations

import getpass
import hashlib

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


def default_name(data_folder: str = "") -> str:
    """One lock per user and data folder: what must never be shared is the database."""
    tag = hashlib.sha1(data_folder.lower().encode()).hexdigest()[:10]
    return f"downloader-{getpass.getuser()}-{tag}"


class SingleInstance(QObject):
    activated = Signal()  # another launch asked us to come to the front

    def __init__(self, name: str | None = None, parent=None):
        super().__init__(parent)
        self._name = name or default_name()
        self._server: QLocalServer | None = None

    def try_lock(self) -> bool:
        """True if this is the only copy. False if another one runs (it has been asked to show)."""
        socket = QLocalSocket()
        socket.connectToServer(self._name)
        if socket.waitForConnected(500):
            socket.write(b"show")
            socket.flush()
            socket.waitForBytesWritten(500)
            socket.disconnectFromServer()
            return False
        QLocalServer.removeServer(self._name)  # a leftover from a copy that crashed
        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._on_connection)
        return self._server.listen(self._name)

    def release(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None

    def _on_connection(self) -> None:
        while self._server is not None and self._server.hasPendingConnections():
            self._server.nextPendingConnection().deleteLater()
            self.activated.emit()
