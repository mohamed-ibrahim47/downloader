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

    def __init__(self):
        super().__init__()
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
            self._task.failed.emit(str(e))
        except Exception as e:  # yt-dlp and friends raise many kinds of errors
            from ..engine.youtube import friendly_error
            self._task.failed.emit(friendly_error(str(e)) if str(e) else "couldn't read this link")
        else:
            self._task.done.emit(result)


def run_in_pool(fn, *args) -> Task:
    task = Task()
    _alive.add(task)
    # Start on the next event-loop turn: the caller connects done/failed right after this
    # returns, and a fast worker must not emit before those connections exist.
    runnable = _Runnable(task, fn, args)
    QTimer.singleShot(0, lambda: QThreadPool.globalInstance().start(runnable))
    return task
