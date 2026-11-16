"""The source bandit: which kind of search finds a discovery card worth downloading (design v5, V2).

Slot 5 of For you shows one card from ranks 11–30. Each refresh brings new videos, so a score per video
can't be learned. The kind of search that found it comes back every time, so that is what this learns:
three "arms", `channel`, `newest` and `words`.

**Each arm's belief** is a Beta distribution over "the chance a discovery card from this arm gets downloaded":

    Beta(a, b),   a = 1 + downloads,   b = 1 + no's

The 1's are the starting belief: before any data every chance from 0 to 1 is equally likely. A download adds
1 to a, a dismissal 1 to b, a card scrolled past 0.3 to b (the same weights training uses).

**Old taste fades.** Each row counts `0.5 ** (age_days / 30)`: a month-old download counts half. Without
it, an arm that was good in spring would stay favoured after the owner's taste moved on (a
"non-stationary" bandit).

**Thompson sampling.** Each draw, every arm that has a video in the pool draws one guess θ from its Beta,
and the arm with the highest guess wins. An arm with little data has a wide Beta: it sometimes guesses high
and gets tried. An arm known to be bad has a narrow, low Beta and almost never wins. Exploring comes from
the uncertainty itself; there is no "explore 10% of the time" knob to tune.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .dataset import Row
from .rank import Suggestion

ARMS = ("channel", "newest", "words")
HALF_LIFE_DAYS = 30.0
PROPENSITY_DRAWS = 200
_DAY = 86400.0


def fade(at: float, now: float) -> float:
    """How much a row from time `at` still counts: 1 now, 1/2 after 30 days, 1/4 after 60."""
    age_days = max(0.0, now - at) / _DAY
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def kind_of_row(x: list[float]) -> str:
    """The search kind from a row's inputs (x[4] is kind_channel, x[5] kind_newest, both 0 is words)."""
    if x[4] == 1:
        return "channel"
    if x[5] == 1:
        return "newest"
    return "words"


def kind_of(s: Suggestion) -> str:
    """A candidate's search kind; None (logged before 1.2.1) counts as words, as in `features.py`."""
    return s.candidate.kind if s.candidate.kind in ARMS else "words"


def discovery_rows(rows: list[Row]) -> list[Row]:
    """Only the rows of discovery cards (input 7, `discovery`, is 1)."""
    return [r for r in rows if r.x[7] == 1]


@dataclass
class Arm:
    name: str
    yes: float  # downloads, faded
    no: float  # dismissals + 0.3 × scrolled past, faded

    @property
    def a(self) -> float:
        return 1.0 + self.yes

    @property
    def b(self) -> float:
        return 1.0 + self.no

    @property
    def rate(self) -> float:
        """The Beta's mean: the arm's best single guess at its download chance."""
        return self.a / (self.a + self.b)


def arm_stats(rows: list[Row], now: float) -> dict[str, Arm]:
    """Each arm's faded yes/no counts from the discovery rows. All three arms are always there."""
    arms = {name: Arm(name, 0.0, 0.0) for name in ARMS}
    for r in discovery_rows(rows):
        w = r.weight * fade(r.at, now)
        arm = arms[kind_of_row(r.x)]
        if r.label == 1:
            arm.yes += w
        else:
            arm.no += w
    return arms


def pick_source(pool: list[Suggestion], arms: dict[str, Arm],
                rng: random.Random) -> tuple[Suggestion, dict[str, float]] | None:
    """One Thompson draw. Returns the card and the guess θ each arm drew, or None for an empty pool.

    Only arms with a video in the pool draw. The winning arm's card is its first in the pool, which is
    in the ranker's order, so it is that arm's best-scored video.
    """
    first: dict[str, Suggestion] = {}
    for s in pool:
        first.setdefault(kind_of(s), s)
    if not first:
        return None
    draws = {name: rng.betavariate(arms[name].a, arms[name].b) for name in ARMS if name in first}
    winner = max(draws, key=draws.get)
    return first[winner], draws


def propensity(pick_fn, pool: list[Suggestion], chosen_id: str, rng: random.Random,
               n: int = PROPENSITY_DRAWS) -> float:
    """The chance `pick_fn` picks `chosen_id` from this pool, estimated by drawing `n` times and counting.

    Logged with each pick for off-policy evaluation later ("how would another mode have done?"), which
    needs the probability of every logged pick. The draws use their own rng, seeded once from `rng`, so
    the caller's sequence moves by one number, not by n draws.
    """
    own = random.Random(rng.random())
    hits = 0
    for _ in range(n):
        got = pick_fn(pool, own)
        if isinstance(got, tuple):
            got = got[0]
        if got is not None and got.candidate.video_id == chosen_id:
            hits += 1
    return hits / n
