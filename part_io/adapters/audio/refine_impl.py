"""Alignment refinement for detected audio matches."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from part_io.adapters.audio.matcher import AudioMatch, anchor_to_onset, cross_correlate_align


class _MatchLike(Protocol):
    start_seconds: float
    end_seconds: float
    score: float


def refine_matches(
    *,
    matches: Sequence[_MatchLike],
    source_path: Path,
    sample_path: Path,
) -> list[AudioMatch]:
    """Refine match positions using onset anchoring then waveform cross-correlation."""
    refined: list[AudioMatch] = []
    for match in matches:
        start = float(match.start_seconds)
        end = float(match.end_seconds)
        as_audio_match = AudioMatch(
            start_seconds=start,
            end_seconds=end,
            duration_seconds=end - start,
            score=float(match.score),
        )
        anchored = anchor_to_onset(match=as_audio_match, source_path=source_path)
        aligned = cross_correlate_align(
            match=anchored,
            source_path=source_path,
            sample_path=sample_path,
        )
        refined.append(aligned)
    return refined


__all__ = ["refine_matches"]
