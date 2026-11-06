"""Which first: score every candidate against the profile, keep the list varied, explain each pick (spec B6)."""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .candidates import Candidate
from .profile import Profile
from .text import tokenize
from .tfidf import cosine, fit_idf, vector, weighted_sum

CHANNEL_BOOST = 0.15  # score added for a channel you like most (affinity 1.0)
MAX_PER_CHANNEL = 2  # a channel appears at most this often in the list
TOP_K = 10
DISCOVERY_POSITION = 5  # 1-based slot for "something a little different"
DISCOVERY_RANKS = (11, 30)  # 1-based, inclusive: where the discovery pick comes from
DISCOVERY_COPY = "Something a little different"


@dataclass
class Suggestion:
    candidate: Candidate
    score: float  # similarity + boost
    similarity: float  # cosine with the profile
    boost: float  # CHANNEL_BOOST × channel affinity
    because: str | None  # title of the download it is most like
    shared: list[str]  # up to 3 words it shares with that download
    discovery: bool = False


def _model(profile: Profile, cands: list[Candidate]):
    """IDF over the profile docs and the candidates, the profile vector, and each candidate's vector."""
    cand_tokens = [tokenize(c.title, channel=c.channel) for c in cands]
    idf = fit_idf([d.tokens for d in profile.docs] + cand_tokens)
    doc_vecs = [vector(d.tokens, idf) for d in profile.docs]
    profile_vec = weighted_sum(zip(doc_vecs, (d.weight for d in profile.docs)))
    return idf, doc_vecs, profile_vec, cand_tokens


def _explain(profile: Profile, doc_vecs: list[dict], cand_vec: dict, cand_tokens: list[str]):
    """The liked download most like this candidate, and the words they share."""
    best, best_cos = None, 0.0
    for doc, vec in zip(profile.docs, doc_vecs):
        if doc.weight <= 0:
            continue
        c = cosine(vec, cand_vec)
        if c > best_cos:
            best, best_cos = doc, c
    if best is None:
        return None, []
    words = [t for t in dict.fromkeys(cand_tokens) if not t.startswith("ch:") and t in best.tokens]
    words.sort(key=lambda w: -cand_vec.get(w, 0.0))  # stable: ties stay in title order
    return best.title, words[:3]


def rank(profile: Profile, cands: list[Candidate], k: int = TOP_K,
         rng: random.Random | None = None) -> list[Suggestion]:
    rng = rng or random.Random()
    idf, doc_vecs, profile_vec, cand_tokens = _model(profile, cands)
    scored = []
    for c, tokens in zip(cands, cand_tokens):
        vec = vector(tokens, idf)
        similarity = cosine(profile_vec, vec)
        boost = CHANNEL_BOOST * profile.channels.get(c.channel, 0.0) if c.channel else 0.0
        because, shared = _explain(profile, doc_vecs, vec, tokens)
        scored.append(Suggestion(c, similarity + boost, similarity, boost, because, shared))
    scored.sort(key=lambda s: (-s.score, s.candidate.video_id))

    # Diversity: skip a channel once it has MAX_PER_CHANNEL places. No channel never counts as one.
    diverse, used = [], {}
    for s in scored:
        ch = s.candidate.channel
        if ch and used.get(ch, 0) >= MAX_PER_CHANNEL:
            continue
        if ch:
            used[ch] = used.get(ch, 0) + 1
        diverse.append(s)

    # Exploration: move one decent-but-not-top pick into slot DISCOVERY_POSITION.
    low, high = DISCOVERY_RANKS
    if len(diverse) >= low:
        pick = rng.choice(diverse[low - 1:high])
        diverse.remove(pick)
        diverse.insert(DISCOVERY_POSITION - 1, replace(pick, discovery=True, because=DISCOVERY_COPY))
    return diverse[:k]


def profile_words(profile: Profile, cands: list[Candidate], n: int = 15) -> list[tuple[str, float]]:
    """The profile's heaviest words, for the "Why this order?" view. Channels are left out."""
    _, _, profile_vec, _ = _model(profile, cands)
    words = [(w, v) for w, v in profile_vec.items() if not w.startswith("ch:")]
    words.sort(key=lambda wv: -wv[1])
    return words[:n]
