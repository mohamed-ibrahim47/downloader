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


def _f(filename, days_ago=0):
    return Download(url="u", kind=Kind.FILE, filename=filename, folder="d", added_at=NOW - days_ago * DAY)


def test_file_download_joins_at_half_weight():
    p = build_profile([_f("blender-4.2.1-windows-x64.msi", 30)], [], NOW)
    assert len(p.docs) == 1
    d = p.docs[0]
    assert d.video_id is None and d.title == "blender" and d.tokens == ["blender"]
    assert round(d.weight, 6) == 0.25  # half of a 30-day-old video's 0.5


def test_documents_never_join():
    p = build_profile([_f("CV_Mohamed_2026.pdf"), _f("notes.docx"), _f("photo.jpg")], [], NOW)
    assert p.docs == []


def test_meaningless_file_name_skipped():
    assert build_profile([_f("setup_x64_v2.exe")], [], NOW).docs == []


def test_files_off_ignores_files():
    p = build_profile([_f("Complete_Python_Bootcamp.zip"), _v("Blank Space")], [], NOW, files=False)
    assert [d.title for d in p.docs] == ["Blank Space"]


def test_file_has_no_channel_affinity():
    p = build_profile([_f("Complete_Python_Bootcamp.zip"), _v("a", channel="TS")], [], NOW)
    assert p.channels == {"TS": 1.0}
