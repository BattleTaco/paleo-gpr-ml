"""Host-level helpers for long-running jobs."""

from __future__ import annotations

import os
import shutil
import subprocess


def inhibit_sleep(why: str) -> subprocess.Popen | None:
    """Block system suspend for as long as the calling process lives.

    The workstation auto-suspends after 15 idle minutes, which silently paused a training run
    for 36 minutes (2026-10-29). This holds a systemd sleep/idle inhibitor tied to our PID, so it
    is released automatically when the job exits or crashes. No-op where systemd-inhibit is
    unavailable (macOS, containers).
    """
    exe = shutil.which("systemd-inhibit")
    if exe is None:
        return None
    try:
        return subprocess.Popen(
            [
                exe,
                "--what=sleep:idle",
                "--who=paleo-gpr-ml",
                f"--why={why}",
                "--mode=block",
                "tail",
                f"--pid={os.getpid()}",
                "-f",
                "/dev/null",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
