"""Putting the PC to sleep when the queue finishes (Windows)."""
from __future__ import annotations

import subprocess


def sleep_pc(runner=subprocess.run) -> None:
    # Note: with hibernation turned on, Windows hibernates instead of sleeping here.
    runner(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=False)
