"""Train the starter word vectors, check v4 on the simulated months, and write
`downloader/engine/recommend/starter_vectors.py` (design v4, E5).

    .venv/Scripts/python scripts/make_embeddings.py TITLES.txt SIM_A.sqlite SIM_B.sqlite [...] [--drop FILE]

1. Train word2vec (embed.train) on the collected public titles plus every simulated month's titles.
2. Check: for each month, rebuild its lists with meaning recomputed from these vectors (as the app does for
   old rows) and score v1 and v4 at each MEANING_MIXES value. Leave one month out: the mix is picked on the
   other months and judged on the one left out. DEFAULT_ON = v4's mean beats v1's.
3. Write the module, plus `starter-words.txt` (every word with its count) next to TITLES for the owner to read.
   `--drop FILE`: words (one per line) to leave out of the vectors, after the owner read the list.

The month databases are copied before use, so the originals are never written to. Titles are never written
into the repo: only words and numbers.
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from downloader.engine.recommend import embed  # noqa: E402
from downloader.engine.recommend.dataset import rows as training_rows  # noqa: E402
from downloader.engine.recommend.embed import Vectors, cosine  # noqa: E402
from downloader.engine.recommend.evaluate import rank_lists  # noqa: E402
from downloader.engine.recommend.service import Recommender  # noqa: E402
from downloader.engine.recommend.text import tokenize  # noqa: E402
from downloader.engine.settings import Settings  # noqa: E402
from downloader.engine.store import Store  # noqa: E402

OUT = ROOT / "downloader" / "engine" / "recommend" / "starter_vectors.py"
MEANING_MIXES = (0.25, 0.5, 0.75)
# Pairs the banner may use as its example: the closest pair (in these vectors) wins.
EXAMPLES = [("piano", "acoustic"), ("cover", "acoustic"), ("guitar", "chords"), ("tutorial", "course"),
            ("lion", "wildlife"), ("recipe", "cooking"), ("python", "programming"), ("trailer", "movie"),
            ("documentary", "wildlife"), ("beginners", "tutorial")]


def month_titles(path: str) -> list[str]:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT title FROM rec_events WHERE title IS NOT NULL UNION "
                          "SELECT title FROM rec_candidates WHERE title IS NOT NULL UNION "
                          "SELECT title FROM downloads WHERE title IS NOT NULL").fetchall()
    finally:
        db.close()
    return [r[0] for r in rows]


def month_scores(path: str, vectors: Vectors, work: Path) -> dict[float, tuple[float, float]]:
    """{mix: (v1 NDCG, v4 NDCG)} for one month, meaning recomputed from `vectors`."""
    copy = work / f"{Path(path).parent.name}.sqlite"
    shutil.copyfile(path, copy)
    store = Store(copy)
    try:
        rec = Recommender(store, search=lambda q, n: [])
        events = rec.log.events()
        meaning = rec._meaning_at(events, vectors, Settings.load(store))
        rows = training_rows(events, open_refresh=-1.0, meaning=meaning)  # a finished month: all settled
        out = {}
        for mix in MEANING_MIXES:
            _n, ndcg, _hit = rank_lists(rows, {"mix": mix})
            out[mix] = (ndcg["v1"], ndcg["v4"])
        return out
    finally:
        store.close()


def check(months: list[str], vectors: Vectors) -> tuple[float, bool, dict]:
    with tempfile.TemporaryDirectory(dir="D:/dev/tmp") as tmp:
        scores = [month_scores(m, vectors, Path(tmp)) for m in months]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    held_v1, held_v4 = [], []
    for i, s in enumerate(scores):  # leave month i out: pick the mix on the others
        others = [x for j, x in enumerate(scores) if j != i]
        mix = max(MEANING_MIXES, key=lambda m: mean([o[m][1] for o in others]))
        held_v1.append(s[mix][0])
        held_v4.append(s[mix][1])
    best = max(MEANING_MIXES, key=lambda m: mean([s[m][1] for s in scores]))
    result = {"v1": round(mean(held_v1), 4), "v4": round(mean(held_v4), 4),
              "per_mix": {m: round(mean([s[m][1] for s in scores]), 4) for m in MEANING_MIXES},
              "months": len(months)}
    return best, result["v4"] > result["v1"], result


def example(v: Vectors) -> tuple[str, str] | None:
    pairs = [(cosine(v.get(a), v.get(b)), a, b) for a, b in EXAMPLES if v.get(a) and v.get(b)]
    pairs = [p for p in pairs if p[0] > 0.5]
    return max(pairs)[1:] if pairs else None


def without(v: Vectors, drop: set[str]) -> Vectors:
    keep = [i for i, w in enumerate(v.words) if w not in drop]
    return Vectors([v.words[i] for i in keep], [v.vecs[i] for i in keep], [v.counts[i] for i in keep])


def write(v: Vectors, mix: float, on: bool, result: dict, ex, n_titles: int, out: Path = OUT) -> None:
    text = f'''"""The starter embeddings: trained on public titles so "match by meaning" works from day one (design v4, E5).

Generated by `scripts/make_embeddings.py` on {datetime.date.today()}. Do not edit by hand; run the script again.
Trained on {n_titles:,} public YouTube titles and the simulated months' titles. Only words and numbers are here.
"""
from __future__ import annotations

WORDS = {chr(10).join(v.words)!r}
COUNTS: list[int] = {v.counts!r}
DIM = {v.dim}
DATA = {v.to_b64()!r}
MEANING_MIX = {mix}
DEFAULT_ON = {on}
CHECK: dict = {result!r}  # mean NDCG@10 on a month the mix was not picked on
EXAMPLE: tuple[str, str] | None = {ex!r}
TRAINED_ON = {{"titles": {n_titles}, "words": {len(v)}}}

_cache = None


def vectors():
    """The starter `embed.Vectors`, or None when none ship."""
    global _cache
    if not WORDS:
        return None
    if _cache is None:
        from .embed import Vectors
        _cache = Vectors.from_b64(WORDS.split("\\n"), COUNTS, DATA, DIM)
    return _cache
'''
    out.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("titles")
    ap.add_argument("months", nargs="+")
    ap.add_argument("--drop", help="a file of words to leave out, one per line")
    ap.add_argument("--epochs", type=int, default=embed.STARTER_EPOCHS)
    ap.add_argument("--subsample", type=float, default=embed.SUBSAMPLE)
    ap.add_argument("--out", default=str(OUT), help="where to write the module (default: the app's)")
    args = ap.parse_args()

    titles = Path(args.titles).read_text(encoding="utf-8").splitlines()
    for m in args.months:
        titles += month_titles(m)
    titles = sorted({t for t in titles if t.strip()})
    print(f"{len(titles):,} distinct titles", flush=True)

    t0 = time.time()
    v = embed.train([tokenize(t) for t in titles], epochs=args.epochs, min_count=embed.STARTER_MIN_COUNT,
                    subsample=args.subsample)
    print(f"trained {len(v):,} words × {v.dim} in {time.time() - t0:.0f} s", flush=True)
    if args.drop:
        drop = {w.strip().lower() for w in Path(args.drop).read_text(encoding="utf-8").splitlines() if w.strip()}
        v = without(v, drop)
        print(f"dropped {len(drop)} words: {len(v):,} left", flush=True)

    for w in ("piano", "cover", "tutorial", "python", "lion", "recipe", "football", "trailer", "عمرو", "طبخ"):
        if v.get(w):
            print(f"  {w} → {', '.join(v.nearest(w, 6))}", flush=True)

    mix, on, result = check(args.months, v)
    ex = example(v)
    print(f"check (held-out months): {result}", flush=True)
    print(f"MEANING_MIX {mix}, DEFAULT_ON {on}, EXAMPLE {ex}", flush=True)

    words_file = Path(args.titles).with_name("starter-words.txt")
    words_file.write_text("".join(f"{w}\t{c}\n" for w, c in zip(v.words, v.counts)), encoding="utf-8", newline="\n")
    write(v, mix, on, result, ex, len(titles), Path(args.out))
    print(f"wrote {args.out} and {words_file}", flush=True)


if __name__ == "__main__":
    main()
