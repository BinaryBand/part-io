"""Non-blocking ffplay playback with a stoppable, position-aware handle.

:func:`~partio.adapters.audio.clips.play_audio_segment` blocks until the clip
ends, which forces a listener to sit through every audition before answering.
This module starts ffplay in the background and hands back a handle the caller
can poll for the playhead position and stop the moment a decision is made.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from partio.adapters.process.runner import resolve_executable

_STOP_TIMEOUT_SECONDS = 2.0


class PlaybackHandle:
    """A running ffplay process for one segment, with a wall-clock playhead.

    ffplay gives no position feedback, so the playhead is derived from elapsed
    monotonic time since launch -- accurate enough to drive a progress bar, and
    it never blocks the caller.
    """

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        start_seconds: float,
        duration_seconds: float,
        started_at: float,
    ) -> None:
        """Wrap a launched ffplay *process* covering a known segment."""
        self._process = process
        self._started_at = started_at
        self.start_seconds = start_seconds
        self.duration_seconds = duration_seconds

    @property
    def elapsed_seconds(self) -> float:
        """Seconds of wall clock since playback started."""
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def position_seconds(self) -> float:
        """Playhead position in source-file seconds, clamped to the segment end."""
        return self.start_seconds + min(self.elapsed_seconds, self.duration_seconds)

    def is_playing(self) -> bool:
        """Whether the segment is still sounding."""
        return self._process.poll() is None and self.elapsed_seconds < self.duration_seconds

    def stop(self) -> None:
        """Terminate playback, escalating to a kill if ffplay ignores the signal."""
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()


def start_audio_segment(
    *, source_path: Path, start_seconds: float, duration_seconds: float
) -> PlaybackHandle:
    """Start playing a segment of *source_path* without blocking.

    ffplay's stdin is detached so it cannot swallow the keystrokes the calling
    TUI is reading.
    """
    command = [
        resolve_executable("ffplay"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nodisp",
        "-autoexit",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        str(source_path),
    ]
    process = subprocess.Popen(  # noqa: S603 - executable resolved to an absolute path above
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return PlaybackHandle(
        process,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        started_at=time.monotonic(),
    )


__all__ = ["PlaybackHandle", "start_audio_segment"]
