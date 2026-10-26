import time

from downloader.engine.ratelimit import TokenBucket


def test_token_bucket_limits_rate():
    bucket = TokenBucket(1_000_000)
    t0 = time.monotonic()
    for _ in range(3):
        bucket.consume(500_000)
    assert time.monotonic() - t0 >= 0.4


def test_token_bucket_none_is_instant():
    bucket = TokenBucket(None)
    t0 = time.monotonic()
    for _ in range(100):
        bucket.consume(10_000_000)
    assert time.monotonic() - t0 < 0.1


def test_set_rate_none_lifts_limit():
    bucket = TokenBucket(1000)
    bucket.set_rate(None)
    t0 = time.monotonic()
    bucket.consume(10_000_000)
    assert time.monotonic() - t0 < 0.1
