"""Everything the user can change, saved in the store as one JSON value."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .scheduler import Schedule
from .store import Store

WHEN_DONE = ("nothing", "notify", "sleep")
CONNECTION_CHOICES = (1, 4, 8, 16)
DISCOVERY_MODES = ("random", "source", "contextual")  # For you: who picks the discovery card (v5)


def _downloads(*sub: str) -> str:
    return str(Path.home().joinpath("Downloads", *sub))


@dataclass
class Settings:
    max_active: int = 3
    connections: int = 8
    theme: str = "dark"
    watch_clipboard: bool = True
    speed_limit_on: bool = False
    speed_limit_kbps: int = 2048
    when_done: str = "nothing"
    folder_general: str = field(default_factory=_downloads)
    folder_video: str = field(default_factory=lambda: _downloads("Video"))
    folder_playlist: str = field(default_factory=lambda: _downloads("Playlists"))
    number_playlist_files: bool = True
    subtitles: bool = False
    suggest_videos: bool = True  # the For you page; off hides it and it never searches
    rec_learned: bool = True  # For you: rank with the learned model (v2) once it beats v1
    rec_trainer: str = "hand"  # "hand" (our gradient descent) or "sklearn" (when it's installed)
    rec_meaning: bool = True  # For you: "Match by meaning" (v4 embeddings)
    rec_files: bool = True  # For you: "Use file downloads for suggestions" (programs, archives, course videos)
    rec_discovery: str = "source"  # For you: "Discovery card", one of DISCOVERY_MODES
    rec_two_tower: bool = False  # For you: "Try the two-tower ranker (v7, experimental)"; off = v7 never runs
    schedule: Schedule = field(default_factory=Schedule)

    @property
    def speed_limit_bps(self) -> int | None:
        """The limit in bytes per second (KB here means 1024 bytes), or None when off."""
        return self.speed_limit_kbps * 1024 if self.speed_limit_on else None

    @classmethod
    def load(cls, store: Store) -> "Settings":
        raw = store.get_setting("settings", {}) or {}
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in known}
        if isinstance(values.get("schedule"), dict):
            sched_known = {f.name for f in fields(Schedule)}
            values["schedule"] = Schedule(**{k: v for k, v in values["schedule"].items() if k in sched_known})
        if values.get("rec_discovery") not in DISCOVERY_MODES:
            values.pop("rec_discovery", None)  # unknown (a newer or broken save): the default
        return cls(**values)

    def save(self, store: Store) -> None:
        store.set_setting("settings", asdict(self))
