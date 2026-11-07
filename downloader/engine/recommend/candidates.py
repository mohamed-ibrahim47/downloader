"""What we could show: search queries from the profile, YouTube search, then a filter (spec B5)."""
from __future__ import annotations

from dataclasses import dataclass

from .. import youtube
from ..errors import DownloadError
from .profile import Profile
from .text import tokenize
from .tfidf import fit_idf, vector

QUERIES_PER_REFRESH = 4  # up to 2 channel names, then word pairs
CHANNEL_QUERIES = 2
RESULTS_PER_QUERY = 15
MIN_DURATION = 60  # seconds; shorter is a Short


@dataclass
class Candidate:
    video_id: str
    url: str
    title: str
    channel: str | None
    duration: int | None
    query: str  # the search that found it, kept for the learning log


def _pair(words: list[str], weight: float, idf: dict[str, float]) -> tuple[str, float] | None:
    """A doc's two best words, in the order they appear, and their summed score."""
    distinct = list(dict.fromkeys(words))  # unique, in order
    if len(distinct) < 2:
        return None
    vec = vector(words, idf)
    scores = {w: weight * vec.get(w, 0.0) for w in distinct}
    best = sorted(distinct, key=lambda w: -scores[w])[:2]  # a stable sort: ties keep doc order
    best.sort(key=distinct.index)
    return " ".join(best), scores[best[0]] + scores[best[1]]


def queries(profile: Profile, blocked: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Up to 2 channel names (never a blocked one), the newest download, then word pairs (spec B5).

    The newest download always gets a search of its own. Without it, every search came from the old
    favourites, so a new interest never reached YouTube (seen in the month-long simulation).
    Pairs come from the title only: tags are often SEO filler, and these words are sent to YouTube.
    """
    liked_channels = [name for name in profile.channels if name not in blocked]
    channels = sorted(liked_channels, key=lambda name: -profile.channels[name])[:CHANNEL_QUERIES]
    liked = [(tokenize(d.title), d.weight) for d in profile.docs if d.weight > 0]
    idf = fit_idf([words for words, _ in liked])
    out = list(channels)
    newest = max(liked, key=lambda wd: wd[1], default=None)  # weight falls with age: the highest is the newest
    if newest and newest[0]:
        fresh = _pair(*newest, idf)
        text = fresh[0] if fresh else newest[0][0]  # a one-word title is searched as that word
        if text not in out:
            out.append(text)
    pairs = [p for p in (_pair(words, weight, idf) for words, weight in liked) if p]
    pairs.sort(key=lambda p: -p[1])
    for text, _score in pairs:
        if len(out) >= QUERIES_PER_REFRESH:
            break
        if text not in out:
            out.append(text)
    return out


def fetch(queries: list[str], search=youtube.search) -> list[Candidate]:
    """Run each query in turn. A failing query is skipped; if all of them fail, we're offline."""
    found: list[Candidate] = []
    failed = 0
    for q in queries:
        try:
            hits = search(q, RESULTS_PER_QUERY)
        except Exception:
            failed += 1
            continue
        found += [Candidate(h.video_id, h.url, h.title, h.channel, h.duration, q) for h in hits]
    if queries and failed == len(queries):
        raise DownloadError("Couldn't reach YouTube.")
    return found


def keep(cands: list[Candidate], downloaded: set[str], dismissed: set[str], blocked: set[str]) -> list[Candidate]:
    """Drop what the owner has, dismissed or blocked, plus live streams and Shorts. First one wins."""
    seen: set[str] = set()
    out = []
    for c in cands:
        if c.video_id in seen or c.video_id in downloaded or c.video_id in dismissed:
            continue
        if c.channel in blocked or c.duration is None or c.duration < MIN_DURATION:
            continue
        seen.add(c.video_id)
        out.append(c)
    return out
