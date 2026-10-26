from downloader.engine.speed import SpeedMeter


def test_rate_over_last_three_seconds():
    m = SpeedMeter()
    for i in range(30):
        m.add(100_000, now=100.0 + i * 0.1)
    assert abs(m.rate(now=103.0) - 1_000_000) < 100_000


def test_rate_drops_to_zero_when_idle():
    m = SpeedMeter()
    m.add(1_000_000, now=100.0)
    assert m.rate(now=110.0) == 0


def test_history_has_60_points():
    m = SpeedMeter()
    m.add(5000, now=100.5)
    h = m.history(now=101.0)
    assert len(h) == 60
    assert h[-2] == 5000 or h[-1] == 5000


def test_eta_none_when_idle():
    m = SpeedMeter()
    assert m.eta(1000, now=100.0) is None


def test_eta_none_when_size_unknown():
    m = SpeedMeter()
    m.add(300_000, now=99.0)
    assert m.eta(None, now=100.0) is None


def test_eta_from_rate():
    m = SpeedMeter()
    for i in range(30):
        m.add(100_000, now=100.0 + i * 0.1)
    assert abs(m.eta(10_000_000, now=103.0) - 10) < 1.5
