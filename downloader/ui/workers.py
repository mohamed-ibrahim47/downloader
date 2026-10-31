"""Runs slow calls (probing a link, asking YouTube) off the GUI thread and reports back by signal."""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from ..engine.errors import DownloadError

_alive: set["Task"] = set()


class Task(QObject):
    """Lives on the GUI thread. It must outlive the queued delivery of its own result, so it is
    let go only from the GUI thread, one event-loop turn after done/failed arrives."""

    done = Signal(object)
    failed = Signal(str)
    # The same, with `tag` first: a receiver that may have moved on compares it instead of relying
    # on QObject.sender(), which Qt doesn't guarantee for queued delivery.
    tagged_done = Signal(object, object)
    tagged_failed = Signal(object, str)

    def __init__(self):
        super().__init__()
        self.tag = None
        self.done.connect(self._release)
        self.failed.connect(self._release)

    @Slot()
    def _release(self, *_args) -> None:
        QTimer.singleShot(0, lambda: _alive.discard(self))


class _Runnable(QRunnable):
    def __init__(self, task: Task, fn, args):
        super().__init__()
        self._task, self._fn, self._args = task, fn, args

    def run(self) -> None:
        try:
            result = self._fn(*self._args)
        except DownloadError as e:
            self._fail(str(e))
        except Exception as e:  # yt-dlp and friends raise many kinds of errors
            from ..engine.youtube import friendly_error
            self._fail(friendly_error(str(e)) if str(e) else "couldn't read this link")
        else:
            # Tagged first: `done` lets the task go.
            self._task.tagged_done.emit(self._task.tag, result)
            self._task.done.emit(result)

    def _fail(self, message: str) -> None:
        self._task.tagged_failed.emit(self._task.tag, message)
        self._task.failed.emit(message)


def run_in_pool(fn, *args) -> Task:
    task = Task()
    _alive.add(task)
    # Start on the next event-loop turn: the caller connects done/failed right after this
    # returns, and a fast worker must not emit before those connections exist.
    runnable = _Runnable(task, fn, args)
    QTimer.singleShot(0, lambda: QThreadPool.globalInstance().start(runnable))
    return task
