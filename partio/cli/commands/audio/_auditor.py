"""Shared interactive auditor for audio review and bootstrap.

Builds an :class:`~partio.core.ports.audio.AuditorFn`.  On a terminal this is
the live audition UI -- a section bar with a moving playhead, answered with a
single keystroke while the clip is still playing.  Off a terminal (pipes, CI,
tests) it degrades to blocking playback plus an ``input()`` prompt, so the same
auditor stays scriptable.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from partio.adapters.audio.clips import play_audio_segment
from partio.cli.commands.audio._audition_ui import AnsweredSpan, run_audition
from partio.cli.output import emit

if TYPE_CHECKING:
    from pathlib import Path

    from partio.core.ports.audio import AuditorFn

# Context shown around the clip when the caller has no search region to frame.
_CONTEXT_SECONDS = 30.0


def build_interactive_auditor(
    *,
    source_path: Path,
    region_start: float | None = None,
    region_end: float | None = None,
) -> AuditorFn:
    """Return an auditor that plays segments and collects a yes/no verdict.

    *region_start* / *region_end* frame the section bar.  When omitted the bar
    falls back to a window centred on whichever clip is being auditioned.
    """
    answered: list[AnsweredSpan] = []

    def _audition(start_seconds: float, duration_seconds: float, question: str) -> bool:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return _prompt_fallback(
                source_path=source_path,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                question=question,
            )

        end_seconds = start_seconds + duration_seconds
        result = run_audition(
            source_path=source_path,
            clip_start=start_seconds,
            clip_duration=duration_seconds,
            question=question,
            region_start=(
                region_start
                if region_start is not None
                else max(0.0, start_seconds - _CONTEXT_SECONDS)
            ),
            region_end=(region_end if region_end is not None else end_seconds + _CONTEXT_SECONDS),
            answered=list(answered),
        )
        if result is None:
            raise KeyboardInterrupt
        answered.append(AnsweredSpan(start=start_seconds, end=end_seconds, answer=result))
        return result

    return _audition


def _prompt_fallback(
    *, source_path: Path, start_seconds: float, duration_seconds: float, question: str
) -> bool:
    """Blocking playback plus a typed y/n answer, for non-TTY use."""
    play_audio_segment(
        source_path=source_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        emit("Please answer y or n.")


__all__ = ["build_interactive_auditor"]
