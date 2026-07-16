"""Shared interactive auditor for audio review and bootstrap.

Builds an :class:`~partio.core.ports.audio.AuditorFn` that plays audio
segments through ``ffplay`` and asks yes/no questions via ``input()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from partio.adapters.audio.clips import play_audio_segment
from partio.cli.output import emit

if TYPE_CHECKING:
    from partio.core.ports.audio import AuditorFn


def build_interactive_auditor(*, source_path: Path) -> AuditorFn:
    """Return an auditor that plays segments and asks yes/no via ``input()``."""

    def _audition(start_seconds: float, duration_seconds: float, question: str) -> bool:
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

    return _audition
