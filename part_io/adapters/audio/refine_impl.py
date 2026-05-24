"""Alignment refinement for detected audio matches."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from part_io.adapters.audio.matcher import AudioMatch, anchor_to_onset


class _MatchLike(Protocol):
    start_seconds: float
    end_seconds: float
    score: float


def align_matches_to_onset(
    *,
    matches: Sequence[_MatchLike],
    source_path: Path,
) -> list[AudioMatch]:
    """Fine-align match positions to the first significant energy onset in the source."""
    aligned: list[AudioMatch] = []
    for match in matches:
        start = float(match.start_seconds)
        end = float(match.end_seconds)
        as_audio_match = AudioMatch(
            start_seconds=start,
            end_seconds=end,
            duration_seconds=end - start,
            score=float(match.score),
        )
        aligned.append(anchor_to_onset(match=as_audio_match, source_path=source_path))
    return aligned


__all__ = ["align_matches_to_onset"]
