# Downloader

A fast download manager for Windows, in the spirit of Internet Download Manager. Paste any link (a file,
a YouTube video or a whole playlist) and Downloader works out what it is and shows the right options.

![Downloader, dark theme](docs/screenshots/dark.png)

![Downloader, light theme](docs/screenshots/light.png)

## Features

- **One smart paste bar.** Files start right away; YouTube videos get quality buttons (1080p, 720p,
  480p, Audio) and the size for each; playlists show the video count, total size and a "choose videos" list.
- **Fast, resumable downloads.** Each file is split into parts fetched over 1, 4, 8 or 16 connections at once.
  Pause and resume, even after closing the app. Servers that limit connections are handled by
  backing off to one connection at a time.
- **YouTube videos and playlists** via [yt-dlp](https://github.com/yt-dlp/yt-dlp), with sound merged
  into the video by a bundled FFmpeg. Videos prefer H.264 so they play in any player. If the quality
  you picked isn't offered, the next lower one is used and the card says so. MP3 is available from
  "more formats".
- **Queue**: choose how many downloads run at the same time; extra ones wait. "Start now" jumps the line.
- **Plain-language errors** ("server refused (403) · link may have expired") with a Retry button.
  Network hiccups are retried three times first.
- **Watches the clipboard**: copy a link anywhere and it appears in the paste bar.
- **Speed limit**, **scheduler** (start at night, stop in the morning, on chosen days) and **what to do
  when the queue finishes** (nothing, a notification, or sleep).
- **Light and dark themes**, and a tray icon that keeps downloads going with the window closed.
- Never overwrites: a name that's taken gets ` (2)`.

## Install

Download `downloader-<version>-windows.zip` from [Releases](../../releases), unzip it anywhere and run
`downloader.exe`. No installer or Python needed.

### From source

Python 3.10 or newer:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m downloader
```

Downloads and settings are kept in `%LOCALAPPDATA%\Downloader`. Set `DOWNLOADER_HOME` to keep them
somewhere else.

### Tests and building the .exe

```bash
pip install -e .[dev]
python -m pytest                     # RUN_NETWORK=1 also runs the tests that use YouTube
powershell -File scripts/build_exe.ps1
```

## Please note

Downloader is for downloading content you have the right to download. Respect the terms of the sites
you use and the rights of the people who made the content.

## Credits

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (Unlicense) for YouTube.
- [FFmpeg](https://ffmpeg.org) via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) for merging
  video and sound.
- [Qt for Python (PySide6)](https://doc.qt.io/qtforpython-6/) (LGPLv3) for the interface.
- [Inter](https://rsms.me/inter/) font by Rasmus Andersson, SIL Open Font License 1.1
  (`downloader/ui/assets/fonts/OFL.txt`).

Released under the MIT License. This is a rewrite of a small PyQt5 downloader I made in 2018.
