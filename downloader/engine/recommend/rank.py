"""Which first: score every candidate against the profile, keep the list varied, explain each pick (spec B6)."""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .candidates import Candidate
from .profile import Profile
from .text import tokenize
from .tfidf import cosine, fit_idf, norm, vector, weighted_sum

CHANNEL_BOOST = 0.15  # score added for a channel you like most (affinity 1.0)
MAX_PER_CHANNEL = 2  # a channel appears at most this often in the list
MAX_PER_QUERY = 3  # cards from one search in the top list, while other searches have something to offer
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


@dataclass
class Model:
    """Everything ranking needs, built once: IDF, each liked download's vector and length, the profile."""
    idf: dict[str, float]
    doc_vecs: list[dict[str, float]]
    doc_norms: list[float]
    profile_vec: dict[str, float]
    profile_norm: float
    cand_tokens: list[list[str]]
    index: dict[str, list[int]]  # word -> the liked downloads that have it


def build_model(profile: Profile, cands: list[Candidate]) -> Model:
    cand_tokens = [tokenize(c.title, channel=c.channel) for c in cands]
    idf = fit_idf([d.tokens for d in profile.docs] + cand_tokens)
    doc_vecs = [vector(d.tokens, idf) for d in profile.docs]
    profile_vec = weighted_sum(zip(doc_vecs, (d.weight for d in profile.docs)))
    index: dict[str, list[int]] = {}
    for i, (doc, vec) in enumerate(zip(profile.docs, doc_vecs)):
        if doc.weight > 0:
            for word in vec:
                index.setdefault(word, []).append(i)
    return Model(idf, doc_vecs, [norm(v) for v in doc_vecs], profile_vec, norm(profile_vec), cand_tokens, index)


def _explain(profile: Profile, m: Model, cand_vec: dict, cand_norm: float, cand_tokens: list[str]):
    """The liked download most like this candidate, and the words they share.

    Only downloads that share a word with the candidate can have a cosine above 0, so only those are scored.
    """
    near = sorted({i for word in cand_vec for i in m.index.get(word, ())})  # doc order: ties go to the first
    best, best_cos = None, 0.0
    for i in near:
        c = cosine(m.doc_vecs[i], cand_vec, m.doc_norms[i], cand_norm)
        if c > best_cos:
            best, best_cos = profile.docs[i], c
    if best is None:
        return None, []
    words = [t for t in dict.fromkeys(cand_tokens) if not t.startswith("ch:") and t in best.tokens]
    words.sort(key=lambda w: -cand_vec.get(w, 0.0))  # stable: ties stay in title order
    return best.title, words[:3]


def rank(profile: Profile, cands: list[Candidate], k: int = TOP_K, rng: random.Random | None = None,
         discovery_id: str | None = None, model: Model | None = None) -> list[Suggestion]:
    """`discovery_id`: keep this video in the discovery slot while it is still in range, so the slot
    doesn't change under the owner after a Dismiss."""
    rng = rng or random.Random()
    m = model or build_model(profile, cands)
    scored = []
    for c, tokens in zip(cands, m.cand_tokens):
        vec = vector(tokens, m.idf)
        vec_norm = norm(vec)
        similarity = cosine(m.profile_vec, vec, m.profile_norm, vec_norm)
        boost = CHANNEL_BOOST * profile.channels.get(c.channel, 0.0) if c.channel else 0.0
        because, shared = _explain(profile, m, vec, vec_norm, tokens)
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

    # Variety across searches: one search (one song, say) gets at most MAX_PER_QUERY of the top k places.
    # It's a soft limit: when the other searches run out, the list is filled from the rest in score order.
    # What is held back stays in the list below the top k, so the discovery pick can still come from it.
    top, rest, per_query = [], [], {}
    for s in diverse:
        q = s.candidate.query
        if len(top) < k and per_query.get(q, 0) < MAX_PER_QUERY:
            per_query[q] = per_query.get(q, 0) + 1
            top.append(s)
        else:
            rest.append(s)
    diverse = top + rest  # if top is short of k, the first of `rest` fill it

    # Exploration: move one decent-but-not-top pick into slot DISCOVERY_POSITION.
    low, high = DISCOVERY_RANKS
    if len(diverse) >= low:
        pool = diverse[low - 1:high]
        # Prefer a pick whose search isn't already at its limit among the cards that stay on show.
        stays = diverse[:k - 1]
        open_pool = [s for s in pool
                     if sum(x.candidate.query == s.candidate.query for x in stays) < MAX_PER_QUERY] or pool
        pick = next((s for s in open_pool if s.candidate.video_id == discovery_id), None) or rng.choice(open_pool)
        diverse.remove(pick)
        diverse.insert(DISCOVERY_POSITION - 1, replace(pick, discovery=True, because=DISCOVERY_COPY))
    return diverse[:k]


def profile_words(profile: Profile, cands: list[Candidate], n: int = 15,
                  model: Model | None = None) -> list[tuple[str, float]]:
    """The profile's heaviest words, for the "Why this order?" view. Channels are left out."""
    m = model or build_model(profile, cands)
    words = [(w, v) for w, v in m.profile_vec.items() if not w.startswith("ch:")]
    words.sort(key=lambda wv: -wv[1])
    return words[:n]
