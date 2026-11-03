from downloader.engine.recommend.text import channel_token, tokenize


def test_drops_stop_noise_short_and_numbers():
    assert tokenize("Taylor Swift - Blank Space (Official Music Video) 4K 2014 a") == \
        ["taylor", "swift", "blank", "space"]


def test_arabic_stays_whole_and_emoji_goes():
    assert tokenize("عمرو دياب - تملي معاك 🔥") == ["عمرو", "دياب", "تملي", "معاك"]


def test_only_noise_gives_nothing():
    assert tokenize("Official Video 4K") == [] and tokenize("🔥🔥") == []


def test_tags_add_words_and_channel_is_one_token():
    assert tokenize("Style", tags=["1989 Era", "pop"], channel=" Taylor Swift ") == \
        ["style", "era", "pop", "ch:taylor swift"]


def test_no_channel_no_token():
    assert channel_token("") is None and channel_token(None) is None
    assert not any(t.startswith("ch:") for t in tokenize("Song", channel=None))
