from downloader.engine.models import Download, Kind
from downloader.engine.recommend.profile import Dismissed, build_profile

NOW = 1_800_000_000.0
DAY = 86400


def _v(title, days_ago=0, channel=None, tags=None, kind=Kind.VIDEO, playlist=None, vid=None):
    return Download(url="u", kind=kind, filename="f", folder="d", title=title, channel=channel, tags=tags,
                    playlist_title=playlist, video_id=vid, added_at=NOW - days_ago * DAY)


def test_recency_halves_every_30_days():
    p = build_profile([_v("a"), _v("b", 30), _v("c", 60)], [], NOW)
    assert [round(d.weight, 3) for d in p.docs] == [1.0, 0.5, 0.25]


def test_files_and_untitled_are_ignored():
    f = Download(url="u", kind=Kind.FILE, filename="x.iso", folder="d", title="Ubuntu")
    assert build_profile([f, _v(None)], [], NOW).docs == []


def test_playlist_items_use_title_and_playlist_title():
    p = build_profile([_v("Blank Space", kind=Kind.PLAYLIST_ITEM, playlist="1989 Deluxe")], [], NOW)
    assert p.docs[0].tokens == ["blank", "space", "deluxe"]


def test_tags_json_is_read_and_bad_json_is_ignored():
    assert build_profile([_v("Song", tags='["pop"]')], [], NOW).docs[0].tokens == ["song", "pop"]
    assert build_profile([_v("Song", tags="not json")], [], NOW).docs[0].tokens == ["song"]


def test_dismissed_join_with_negative_weight():
    p = build_profile([_v("a")], [Dismissed("v1", "piano cover", None)], NOW)
    assert p.docs[-1].weight == -0.5 and p.docs[-1].tokens == ["piano", "cover"]


def test_channel_affinity_scaled_to_one():
    p = build_profile([_v("a", channel="TS"), _v("b", channel="TS"), _v("c", channel="Other")], [], NOW)
    assert p.channels == {"TS": 1.0, "Other": 0.5}


def test_ready_needs_three_downloads():
    assert not build_profile([_v("a"), _v("b")], [Dismissed("x", "c", None)], NOW).ready
    assert build_profile([_v("a"), _v("b"), _v("c")], [], NOW).ready
