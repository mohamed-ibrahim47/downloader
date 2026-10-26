import gc

from PySide6.QtCore import QObject, Slot

from downloader.engine.errors import DownloadError
from downloader.ui import workers
from downloader.ui.workers import run_in_pool


class Receiver(QObject):
    def __init__(self):
        super().__init__()
        self.got = []

    @Slot(object)
    def on_done(self, value):
        self.got.append(value)

    @Slot(str)
    def on_failed(self, message):
        self.got.append(message)


def _fail():
    raise DownloadError("nope")


def test_results_always_delivered_even_after_gc(qtbot):
    # Handlers are connected after run_in_pool returns, exactly as callers do; a worker
    # that finishes first must not lose its result.
    receivers = []
    for i in range(200):
        r = Receiver()
        task = run_in_pool(_fail if i % 2 else (lambda v=i: v))
        task.done.connect(r.on_done)
        task.failed.connect(r.on_failed)
        receivers.append(r)
        del task
    gc.collect()
    qtbot.waitUntil(lambda: all(r.got for r in receivers), timeout=5000)
    assert receivers[1].got == ["nope"]
    qtbot.waitUntil(lambda: not workers._alive, timeout=2000)
