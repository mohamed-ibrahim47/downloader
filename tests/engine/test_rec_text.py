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


def test_numbering_and_filler_words_are_dropped():
    # From the month-long simulation: these topped the profile without saying anything about taste.
    assert tokenize("Chopin - Nocturne Op. 9 No. 2") == ["chopin", "nocturne"]
    assert tokenize("Planet Earth II - Part 3 | Special Episode") == ["planet", "earth"]
    assert tokenize("But what is a neural network? Chapter 1") == ["neural", "network"]
