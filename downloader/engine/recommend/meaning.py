"""How close a video is in meaning to what the owner downloads, from our embeddings (design v4, E3).

Each download's title becomes one vector (embed.Vectors.title_vector). A candidate is compared with all of
them, and its meaning score is the mean of its 3 best matches, each times that download's recency weight.
Best matches, not one averaged profile: the average of music, animals and coding means nothing, while a
guitar cover matches the music downloads very well.
"""
from __future__ import annotations

from .embed import Vectors
from .profile import Profile
from .rank import CHANNEL_BOOST

MEANING_TOP = 3


class MeaningIndex:
    def __init__(self, profile: Profile, vectors: Vectors, idf: dict[str, float]):
        self.vectors = vectors
        self.idf = idf
        self._docs: list[tuple[list[float], float]] = []
        self._memo: dict[tuple[str, ...], float] = {}  # one candidate is scored at every draw
        for d in profile.docs:
            v = vectors.title_vector(d.tokens, idf)
            if v is not None and d.weight != 0:
                self._docs.append((v, d.weight))

    def score(self, tokens: list[str]) -> float:
        """Mean of the best MEANING_TOP positive matches with liked downloads, plus the strongest pull-down
        from a dismissed title close to it. 0 when the title has no known word."""
        key = tuple(tokens)
        if key not in self._memo:
            self._memo[key] = self._score(tokens)
        return self._memo[key]

    def _score(self, tokens: list[str]) -> float:
        v = self.vectors.title_vector(tokens, self.idf)
        if v is None:
            return 0.0
        liked, against = [], 0.0
        for dv, w in self._docs:
            c = sum(a * b for a, b in zip(v, dv))  # title vectors have length 1: the dot product is the cosine
            if w > 0:
                if c * w > 0:
                    liked.append(c * w)
            elif c > 0:
                against = min(against, c * w)
        best = sorted(liked, reverse=True)[:MEANING_TOP]
        return (sum(best) / len(best) if best else 0.0) + against


def v4_score(similarity: float, meaning: float, affinity: float, mix: float) -> float:
    """v1's formula with meaning mixed in: at mix 0 it is exactly v1."""
    return (1 - mix) * similarity + mix * meaning + CHANNEL_BOOST * affinity
