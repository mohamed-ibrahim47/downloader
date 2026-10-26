import uuid

from PySide6.QtNetwork import QLocalServer

from downloader.ui.single_instance import SingleInstance


def _pair(tmp_path):
    return f"downloader-test-{uuid.uuid4().hex}", str(tmp_path / "downloader.lock")


SECOND_COPY = """
import sys
from PySide6.QtCore import QCoreApplication
from downloader.ui.single_instance import SingleInstance
app = QCoreApplication([])
print(SingleInstance(sys.argv[1], sys.argv[2]).try_lock())
"""


def test_second_launch_hands_over_to_the_first(qtbot, tmp_path):
    # Review I7: a second copy (a real second process) must not open the same downloads;
    # it asks the first to show its window, then gives up.
    import subprocess
    import sys
    name, lock = _pair(tmp_path)
    first = SingleInstance(name, lock)
    assert first.try_lock() is True
    with qtbot.waitSignal(first.activated, timeout=10000):
        proc = subprocess.Popen([sys.executable, "-c", SECOND_COPY, name, lock],
                                stdout=subprocess.PIPE, text=True)
        qtbot.waitUntil(lambda: proc.poll() is not None, timeout=10000)
    assert proc.stdout.read().strip() == "False"
    first.release()


def test_lock_is_free_again_after_release(qtbot, tmp_path):
    name, lock = _pair(tmp_path)
    first = SingleInstance(name, lock)
    assert first.try_lock()
    first.release()
    again = SingleInstance(name, lock)
    assert again.try_lock() is True
    again.release()


def test_someone_else_holding_the_pipe_name_cannot_stop_the_app(qtbot, tmp_path):
    # Security review: the decision rests on a lock file in the user's own data folder,
    # not on who owns the pipe name.
    name, lock = _pair(tmp_path)
    squatter = QLocalServer()
    assert squatter.listen(name)
    app = SingleInstance(name, lock)
    assert app.try_lock() is True
    app.release()
    squatter.close()
