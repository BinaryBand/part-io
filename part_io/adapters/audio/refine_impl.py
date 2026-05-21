"""Optional experimental refine implementation.

This file is intentionally detachable. The caller must tolerate this module not
being present and fall back to baseline match positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from part_io.adapters.audio.matcher import AudioMatch, anchor_to_onset, cross_correlate_align


def refine_matches(
    *,
    matches: Sequence[AudioMatch],
    source_path: Path,
    sample_path: Path,
) -> list[AudioMatch]:
    """Refine matches using onset anchoring then cross-correlation alignment."""
    refined: list[AudioMatch] = []
    for match in matches:
        anchored = anchor_to_onset(match=match, source_path=source_path)
        aligned = cross_correlate_align(
            match=anchored,
            source_path=source_path,
            sample_path=sample_path,
        )
        refined.append(aligned)
    return refined


__all__ = ["refine_matches"]
