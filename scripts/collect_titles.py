"""Collect public YouTube titles to train the starter embeddings (design v4, E1). Run offline, by hand.

    .venv/Scripts/python scripts/collect_titles.py [--limit N] [--out FILE]

Searches YouTube with the same flat search For you uses (titles only, nothing is downloaded), RESULTS per
query, with a pause between searches. Writes one title per line to
`D:/dev/tmp/rec-titles/titles-YYYYMMDD.txt`. That file stays local: titles are never committed. Only the
word list and the vectors trained from it (`starter_vectors.py`) go into the repo, after the owner reads them.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from downloader.engine import youtube  # noqa: E402

RESULTS = 100  # 215 queries × 100 ≈ 20k titles after duplicates
PAUSE = 2.0  # seconds between searches

# Broad, everyday topics, so common pairs ("cover" / "acoustic", "tutorial" / "course") show up many times.
TOPICS = {
    "music": ["piano cover", "acoustic guitar cover", "live concert", "lofi beats", "jazz piano", "classical music",
              "violin cover", "drum cover", "rock band live", "pop songs playlist", "hip hop beat", "rap freestyle",
              "music theory lesson", "how to sing", "guitar chords for beginners", "piano tutorial easy",
              "synthwave mix", "orchestra symphony", "opera aria", "country music", "reggae", "metal guitar solo",
              "blues guitar", "unplugged session", "music production tutorial", "film score soundtrack",
              "karaoke version", "remix", "a cappella", "choir"],
    "film": ["movie trailer", "movie review", "film analysis", "behind the scenes", "best movie scenes",
             "tv series recap", "anime opening", "documentary full", "short film", "film director interview",
             "how movies are made", "visual effects breakdown", "movie explained ending", "sitcom bloopers",
             "classic movies", "horror movie review", "animated short film", "box office"],
    "tech": ["python tutorial", "javascript course", "learn programming beginners", "data structures algorithms",
             "machine learning explained", "neural network from scratch", "system design interview",
             "linux command line", "git tutorial", "react tutorial", "sql database tutorial", "docker tutorial",
             "rust programming", "c++ tutorial", "coding interview", "computer science lecture",
             "web development course", "android app development", "game development unity", "blender 3d modeling",
             "photoshop tutorial", "excel tips", "video editing tutorial", "kubernetes explained",
             "cybersecurity basics", "build a pc", "smartphone review", "laptop review", "tech news",
             "artificial intelligence", "deep learning lecture", "statistics explained", "cloud computing"],
    "science": ["calculus lecture", "linear algebra", "probability explained", "physics explained", "quantum mechanics",
                "astronomy documentary", "space exploration", "black holes", "chemistry experiments", "biology lecture",
                "evolution explained", "math olympiad problem", "number theory", "geometry proof", "climate science",
                "how the brain works", "history of science", "engineering explained"],
    "nature": ["wildlife documentary", "lions hunting", "ocean animals", "birds of prey", "national geographic",
               "african safari", "big cats", "whales", "insects macro", "rainforest", "dog training", "funny cats",
               "horse riding", "aquarium setup", "nature sounds", "volcano eruption", "deep sea creatures",
               "penguins", "elephants", "sharks"],
    "life": ["easy recipes", "cooking pasta", "baking bread", "street food", "healthy breakfast", "coffee brewing",
             "home workout", "yoga for beginners", "running tips", "football highlights", "basketball highlights",
             "tennis match", "chess strategy", "formula 1", "boxing", "travel vlog", "travel guide japan",
             "budget travel", "camping", "hiking trail", "diy home repair", "woodworking", "gardening tips",
             "car review", "motorcycle", "minecraft", "gameplay walkthrough", "speedrun", "board games",
             "learn english", "learn spanish", "learn arabic", "study with me", "productivity tips",
             "personal finance", "investing basics", "history documentary", "ancient egypt", "world war 2",
             "philosophy explained", "book summary", "podcast interview", "stand up comedy", "magic tricks",
             "drawing tutorial", "painting landscape", "fashion", "makeup tutorial", "photography tips",
             "architecture", "interior design"],
    "arabic": ["أغاني عربية", "عمرو دياب", "أم كلثوم", "فيروز", "محمد منير", "عود عزف", "تلاوة القرآن",
               "أناشيد", "مسلسلات رمضان", "أفلام مصرية", "كرتون عربي", "كرة القدم ملخص", "الدوري المصري",
               "طبخ سهل", "وصفات حلويات", "تعلم البرمجة", "دورة بايثون", "شرح الرياضيات", "فيزياء شرح",
               "تاريخ مصر", "وثائقي حيوانات", "سفر وسياحة", "مراجعة هاتف", "تعلم الانجليزية", "تنمية بشرية",
               "قصص قبل النوم", "كوميديا", "برامج حوارية", "ألعاب فيديو", "تصميم جرافيك", "ذكاء اصطناعي",
               "فوتوشوب شرح", "اكسل شرح", "رياضة منزلية", "قهوة", "سيارات", "طبيعة", "فضاء", "كتب ملخص",
               "شعر عربي", "مهرجانات", "موسيقى هادئة", "رسم", "تصوير", "اقتصاد"],
}
QUERIES = [q for qs in TOPICS.values() for q in qs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries (a quick try)")
    ap.add_argument("--start", type=int, default=1, help="resume at query N, adding to the existing file")
    ap.add_argument("--out", default=f"D:/dev/tmp/rec-titles/titles-{datetime.date.today():%Y%m%d}.txt")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Arabic queries on a Windows console
    queries = QUERIES[:args.limit] if args.limit else QUERIES
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    titles: list[str] = []
    if args.start > 1 and out.exists():  # resuming: keep what is there (ids are not saved, so the
        titles = out.read_text(encoding="utf-8").splitlines()  # same video may come back once)
    for i, q in enumerate(queries, 1):
        if i < args.start:
            continue
        try:
            hits = youtube.search(q, RESULTS)
        except Exception as e:  # one failed search shouldn't lose the rest
            print(f"[{i}/{len(queries)}] {q!r}: failed ({e})", flush=True)
            hits = []
        new = [h.title for h in hits if h.video_id not in seen]
        seen.update(h.video_id for h in hits)
        titles += new
        print(f"[{i}/{len(queries)}] {q!r}: {len(new)} new, {len(titles)} total", flush=True)
        out.write_text("\n".join(titles) + "\n", encoding="utf-8", newline="\n")  # saved as it goes
        time.sleep(PAUSE)
    print(f"{len(titles)} titles from {len(queries)} queries -> {out}")


if __name__ == "__main__":
    main()
