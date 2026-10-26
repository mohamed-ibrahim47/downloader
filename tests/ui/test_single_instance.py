import uuid

from downloader.ui.single_instance import SingleInstance


def test_second_launch_hands_over_to_the_first(qtbot):
    # Review I7: a second copy must not open the same downloads; it raises the first window.
    name = f"downloader-test-{uuid.uuid4().hex}"
    first = SingleInstance(name)
    assert first.try_lock() is True
    second = SingleInstance(name)
    with qtbot.waitSignal(first.activated, timeout=3000):
        assert second.try_lock() is False
    first.release()


def test_lock_is_free_again_after_release(qtbot):
    name = f"downloader-test-{uuid.uuid4().hex}"
    first = SingleInstance(name)
    assert first.try_lock()
    first.release()
    again = SingleInstance(name)
    assert again.try_lock() is True
    again.release()
