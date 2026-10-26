"""YouTube videos and playlists through yt-dlp, with ffmpeg (bundled) to merge sound into video."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import CONNECTION_LOST, message_for_status
from .models import Download, Status
from .ratelimit import TokenBucket

QUALITIES = ("1080p", "720p", "480p", "Audio")
MB_PER_MINUTE = {"1080p": 11.2, "720p": 8.4, "480p": 4.6, "Audio": 1.0}
MEDIA_EXTS = (".mp4", ".m4a", ".mp3", ".webm", ".mkv", ".opus")


@dataclass
class VideoInfo:
    url: str
    title: str
    channel: str
    duration: int | None
    formats: list[dict]


@dataclass
class PlaylistEntry:
    index: int
    url: str
    title: str
    duration: int | None


@dataclass
class PlaylistInfo:
    url: str
    title: str
    entries: list[PlaylistEntry]


@dataclass
class FormatChoice:
    selector: str
    height: int | None
    size: int | None
    note: str | None
    ext: str
    audio_format: str | None = None


@dataclass
class FormatOption:
    id: str
    quality: str
    kind: str
    size: int | None
    selector: str
    audio_format: str | None


# ---- reading what YouTube offers ----------------------------------------------

def fetch_info(url: str) -> VideoInfo | PlaylistInfo:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist", "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") == "playlist":
        entries = []
        for i, e in enumerate(info.get("entries") or [], start=1):
            if not e:
                continue
            entry_url = e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}"
            entries.append(PlaylistEntry(i, entry_url, e.get("title") or f"Video {i}", _int(e.get("duration"))))
        return PlaylistInfo(url=url, title=info.get("title") or "Playlist", entries=entries)
    return VideoInfo(url=info.get("webpage_url") or url, title=info.get("title") or "Video",
                     channel=info.get("channel") or info.get("uploader") or "",
                     duration=_int(info.get("duration")), formats=info.get("formats") or [])


def _int(v) -> int | None:
    return int(v) if isinstance(v, (int, float)) else None


def _size(f: dict) -> int | None:
    return f.get("filesize") or f.get("filesize_approx")


def _is_video(f: dict) -> bool:
    return f.get("vcodec") not in (None, "none") and bool(f.get("height"))


def _is_audio_only(f: dict) -> bool:
    return f.get("vcodec") in (None, "none") and f.get("acodec") not in (None, "none")


def _best_audio(formats: list[dict]) -> dict | None:
    audio = [f for f in formats if _is_audio_only(f)]
    m4a = [f for f in audio if f.get("ext") == "m4a"]
    pool = m4a or audio
    return max(pool, key=lambda f: f.get("abr") or 0) if pool else None


def _best_video_at(formats: list[dict], height: int) -> dict | None:
    at = [f for f in formats if _is_video(f) and f["height"] == height]
    if not at:
        return None
    return max(at, key=lambda f: (str(f.get("vcodec", "")).startswith("avc1"), f.get("ext") == "mp4",
                                  f.get("acodec") in (None, "none"), _size(f) or 0))


def _sum(*sizes: int | None) -> int | None:
    known = [s for s in sizes if s]
    return sum(known) if known else None


def choose_format(formats: list[dict], quality: str) -> FormatChoice:
    audio = _best_audio(formats)
    heights = sorted({f["height"] for f in formats if _is_video(f)})
    if quality == "Audio" or not heights:
        return FormatChoice(selector_for("Audio"), None, _size(audio) if audio else None,
                            None if quality == "Audio" else "no video offered · got sound only", "m4a")
    target = int(quality.rstrip("p"))
    lower = [h for h in heights if h <= target]
    height = max(lower) if lower else min(heights)
    video = _best_video_at(formats, height)
    note = None if height == target else f"{target}p not offered · got {height}p"
    return FormatChoice(selector_for(f"{height}p"), height,
                        _sum(_size(video), _size(audio) if audio else None), note, "mp4")


def selector_for(quality: str) -> str:
    """A yt-dlp format string for a quality button; lower heights are taken when needed."""
    if quality == "Audio":
        return "bestaudio[ext=m4a]/bestaudio"
    h = int(quality.rstrip("p"))
    # H.264 first so the file plays in any player; then any MP4, then whatever exists.
    return (f"bestvideo[height<={h}][vcodec^=avc1]+bestaudio[ext=m4a]"
            f"/bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]"
            f"/bestvideo[height<={h}]+bestaudio/best[height<={h}]/best")


def format_options(formats: list[dict]) -> list[FormatOption]:
    """Every choice for the full format list (board D): each video height, then sound only."""
    audio = _best_audio(formats)
    audio_size = _size(audio) if audio else None
    options = []
    for h in sorted({f["height"] for f in formats if _is_video(f)}, reverse=True):
        video = _best_video_at(formats, h)
        container = {"mp4": "MP4", "webm": "WebM"}.get(video.get("ext"), str(video.get("ext", "")).upper())
        options.append(FormatOption(f"v{h}", f"{h}p", container, _sum(_size(video), audio_size),
                                    f"bestvideo[height={h}]+bestaudio/best[height={h}]", None))
    if audio:
        kbps = f"{round(audio.get('abr') or 128)}k"
        options.append(FormatOption("m4a", kbps, "M4A", audio_size, "bestaudio[ext=m4a]/bestaudio", None))
        options.append(FormatOption("mp3", "192k", "MP3 · converted after download",
                                    int(audio_size * 1.5) if audio_size else None, "bestaudio/best", "mp3"))
    return options


def estimate_playlist_size(entries: list[PlaylistEntry], quality: str) -> int | None:
    seconds = [e.duration for e in entries if e.duration]
    if not seconds:
        return None
    return round(sum(seconds) / 60 * MB_PER_MINUTE[quality] * 1_000_000)


def friendly_error(raw: str) -> str:
    text = re.sub(r"^\s*ERROR:\s*", "", str(raw)).strip()
    m = re.search(r"HTTP Error (\d{3})", text)
    if m:
        return message_for_status(int(m.group(1)))
    lowered = text.lower()
    if "video unavailable" in lowered or "private video" in lowered:
        return "this video is unavailable or private"
    if "sign in" in lowered:
        return "YouTube asks for a sign-in to watch this video"
    if any(k in lowered for k in ("urlopen error", "getaddrinfo", "timed out", "connection")):
        return CONNECTION_LOST
    first = text.splitlines()[0] if text else "unknown problem"
    first = re.sub(r"^\[[^\]]+\]\s*[\w-]+:\s*", "", first)
    return f"YouTube error: {first}"


# ---- downloading -------------------------------------------------------------

class _Paused(Exception):
    pass


class YouTubeJob:
    """Same shape as HttpJob: start(), pause(), run(); reports through on_progress/on_finish."""

    def __init__(self, download: Download, limiter: TokenBucket,
                 on_progress: Callable[[Download], None], on_finish: Callable[[Download], None],
                 backoff: tuple[float, ...] = (1, 2, 4)):
        self.download = download
        self._limiter = limiter
        self._on_progress = on_progress
        self._on_finish = on_finish
        self._backoff = backoff
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self._base = 0
        self._last_progress = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name=f"youtube-{self.download.id}", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join()

    def _options(self) -> dict:
        import imageio_ffmpeg

        d = self.download
        stem = Path(d.filename).stem
        opts = {
            "quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
            "format": d.format_selector or "best",
            "outtmpl": str(Path(d.folder) / f"{stem}.%(ext)s"),
            "merge_output_format": "mp4",
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
            "continuedl": True,
            "progress_hooks": [self._hook],
            "windowsfilenames": True,
        }
        if self._limiter.rate:
            opts["ratelimit"] = self._limiter.rate
        if d.subtitles:
            opts.update(writesubtitles=True, subtitleslangs=["en", "ar"])
        if d.audio_format == "mp3":
            opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                                       "preferredquality": "192"}]
        return opts

    def _hook(self, status: dict) -> None:
        if self._pause.is_set():
            raise _Paused()
        d = self.download
        current = status.get("downloaded_bytes") or 0
        if status.get("status") == "finished":
            # Video and sound come as separate streams; keep a running total across them.
            self._base += status.get("total_bytes") or current
            current = 0
        d.downloaded = self._base + current
        d.size = max(d.size or 0, d.downloaded) or None
        now = time.monotonic()
        if now - self._last_progress >= 0.1:
            self._last_progress = now
            self._on_progress(d)

    def run(self) -> None:
        import yt_dlp

        d = self.download
        d.status, d.error = Status.DOWNLOADING, None
        self._base = 0
        error = None
        for attempt in range(len(self._backoff) + 1):
            try:
                Path(d.folder).mkdir(parents=True, exist_ok=True)
                with yt_dlp.YoutubeDL(self._options()) as ydl:
                    ydl.download([d.url])
                error = None
                break
            except Exception as e:  # yt-dlp may wrap our _Paused, so check the flag first
                if self._pause.is_set():
                    break
                error = friendly_error(str(e))
                retryable = error.startswith("connection lost") or "server error" in error
                if not retryable or attempt == len(self._backoff):
                    break
                time.sleep(self._backoff[attempt])
                self._base = 0

        if self._pause.is_set():
            d.status = Status.PAUSED
        elif error:
            d.status, d.error = Status.FAILED, error
        else:
            self._finish_names()
            d.status, d.finished_at = Status.COMPLETED, time.time()
        self._on_finish(d)

    def _finish_names(self) -> None:
        d = self.download
        stem = Path(d.filename).stem
        found = [p for p in Path(d.folder).glob(f"{_glob_escape(stem)}.*") if p.suffix.lower() in MEDIA_EXTS]
        if found:
            final = max(found, key=lambda p: p.stat().st_mtime)
            d.filename = final.name
            d.size = final.stat().st_size
            d.downloaded = d.size


def _glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)
