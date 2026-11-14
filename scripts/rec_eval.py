"""Print how the For you rankers do on a database: the owner's, or a simulation's (design v2, C5).

    .venv/Scripts/python scripts/rec_eval.py                 # the app's own database
    .venv/Scripts/python scripts/rec_eval.py path/to/db.sqlite
    .venv/Scripts/python scripts/rec_eval.py path/to/db.sqlite --activation tanh --hidden 16

`--activation` (relu, leaky_relu, tanh, sigmoid) and `--hidden` change v3's network, to compare them on the
same rows. The app always uses the defaults in `recommend/neural.py`.

The database is opened read-only; nothing is written or trained into it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downloader.engine.paths import data_home  # noqa: E402
from downloader.engine.recommend import evaluate  # noqa: E402
from downloader.engine.recommend.dataset import rows as training_rows  # noqa: E402
from downloader.engine.recommend.features import NAMES  # noqa: E402
from downloader.engine.recommend.learn import available_trainers, trainer  # noqa: E402
from downloader.engine.recommend.neural import ACTIVATION, ACTIVATIONS, HIDDEN, NetTrainer  # noqa: E402


def _read(path: Path) -> tuple[list[tuple], float | None]:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols = {row[1] for row in db.execute("PRAGMA table_info(rec_events)")}
        if "refresh" not in cols:
            sys.exit("This database hasn't been opened by version 1.2.1 or later yet, so it has nothing v2 "
                     "can learn from. Rows logged before 1.2.1 lack the refresh and search kind.")
        events = db.execute("SELECT id, video_id, event, features, at, refresh FROM rec_events "
                            "WHERE event IN ('shown', 'downloaded', 'dismissed') ORDER BY id").fetchall()
        row = db.execute("SELECT value FROM settings WHERE key = 'rec_last_refresh'").fetchone()
        return events, json.loads(row[0]) if row else None
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="How the For you rankers do on a database.")
    ap.add_argument("db", nargs="?", type=Path, default=data_home() / "downloader.db")
    ap.add_argument("--activation", choices=sorted(ACTIVATIONS), default=ACTIVATION)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    args = ap.parse_args()
    path = args.db
    events, last = _read(path)
    rows = training_rows(events, open_refresh=last)
    lists = len({r.refresh for r in rows})
    print(f"{path}\n{len(rows)} labelled rows in {lists} lists: {sum(r.label for r in rows)} downloaded, "
          f"{sum(1 for r in rows if not r.label and r.weight == 1.0)} dismissed, "
          f"{sum(1 for r in rows if r.weight < 1.0)} scrolled past\n")
    for name in available_trainers():
        started = time.time()
        rep = evaluate.evaluate(rows, trainer(name), NetTrainer(hidden=args.hidden, activation=args.activation))
        took = time.time() - started
        ranker, why = evaluate.choose(rep)
        print(f"== trainer: {name} (v2) · v3: {args.hidden} hidden, {args.activation}")
        print(f"   ranks the page: {ranker}: {why}")
        if rep.weights is None:
            print("   (nothing to train on yet)\n")
            continue
        print(f"   test lists with a download: {rep.test_lists}")
        print(f"   {'ranker':<10}{'NDCG@10':>9}{'hit@3':>8}")
        for r in evaluate.RANKERS:
            print(f"   {r:<10}{rep.ndcg.get(r, 0):>9.3f}{rep.hit3.get(r, 0):>8.2f}")
        print(f"   v2 AUC {rep.auc:.3f}, log-loss {rep.log_loss:.3f}")
        if rep.net is not None:
            print(f"   v3 AUC {rep.v3_auc:.3f}, log-loss {rep.v3_log_loss:.3f}, {rep.net.size()} weights, "
                  f"stopped after {rep.net.epochs} passes ({took:.1f} s to train both)")
        print("   weights (standardised, trained on the older 80% of lists):")
        for feature, w in zip(NAMES, rep.weights.w):
            print(f"     {feature:<14}{w:+.3f}")
        print(f"     {'bias':<14}{rep.weights.b:+.3f}\n")


if __name__ == "__main__":
    main()
