"""Turn a video's title, tags and channel into the words the recommender compares."""
from __future__ import annotations

import re
from typing import Iterable

# Common words that say nothing about taste.
STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by", "with", "from", "it", "off",
    "is", "my", "your", "you", "me", "this", "that", "but", "what", "how", "why", "la",
    "في", "من", "على", "عن", "مع", "و", "يا", "ال",
    # Numbering and filler, found topping the profile in the month-long simulation
    # ("Nocturne Op. 9 No. 2", "Planet Earth II - Part 3 | Special Episode"). A one-time list: the long
    # tail is left to TF-IDF now, and to learned embeddings (v4) later.
    "no", "op", "ii", "iii", "iv", "part", "ep", "episode", "special", "vol", "chapter",
})

# Words that describe the upload, not the video (spec B3).
NOISE_WORDS = frozenset({
    "official", "video", "audio", "lyrics", "lyric", "live", "hd", "4k", "1080p", "full", "ft", "feat",
    "music", "mv", "remastered",
})

# \w is Unicode-aware: it keeps Arabic letters whole and drops emoji and punctuation.
_WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower())
            if len(w) >= 2 and not w.isdigit() and w not in STOP_WORDS and w not in NOISE_WORDS]


def channel_token(channel: str | None) -> str | None:
    """A channel is one token, never split into words, so it is matched as a channel."""
    name = (channel or "").strip().lower()
    return "ch:" + name if name else None


# Words in a file name that describe the package, not what it is (design v4, E6).
PACKAGING_WORDS = frozenset({
    "setup", "installer", "install", "portable", "full", "final", "release", "latest", "windows", "win",
    "img", "dsc", "copy", "new",
})
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_FILE_SPLIT = re.compile(r"[\s_\-.+]+")


def file_words(filename: str) -> list[str]:
    """The words of a file name that may say what it is: `blender-4.2.1-windows-x64.msi` → ["blender"].

    Drops the extension, versions and architectures (any part with a digit), packaging words, and words
    under 3 letters. An empty list means the name says nothing and the file is skipped.
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    out = []
    for part in _FILE_SPLIT.split(stem):
        for w in _CAMEL.sub(" ", part).lower().split():
            if (len(w) >= 3 and not any(ch.isdigit() for ch in w) and w not in PACKAGING_WORDS
                    and w not in STOP_WORDS and w not in NOISE_WORDS):
                out.append(w)
    return out


def tokenize(title: str, tags: Iterable[str] = (), channel: str | None = None) -> list[str]:
    """Title words, then tag words, then the channel token."""
    tokens = _words(title)
    for tag in tags:
        tokens += _words(tag)
    ch = channel_token(channel)
    if ch:
        tokens.append(ch)
    return tokens
