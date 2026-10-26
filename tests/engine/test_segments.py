from downloader.engine.models import Segment
from downloader.engine.segments import plan_segments


def test_segments_cover_file_contiguously():
    segs = plan_segments(10_000_000, 8, True)
    assert len(segs) == 8
    assert segs[0].start == 0
    assert segs[-1].end == 9_999_999
    assert all(a.end + 1 == b.start for a, b in zip(segs, segs[1:]))
    assert [s.index for s in segs] == list(range(8))


def test_small_file_one_segment():
    assert plan_segments(500_000, 8, True) == [Segment(0, 0, 499_999)]


def test_segment_count_limited_by_min_part():
    assert len(plan_segments(3_500_000, 8, True)) == 3


def test_not_resumable_one_segment():
    assert plan_segments(10_000_000, 8, False) == [Segment(0, 0, 9_999_999)]


def test_unknown_size():
    assert plan_segments(None, 8, True) == [Segment(0, 0, -1)]
