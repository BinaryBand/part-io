"""Pure cut-plan helpers for ad removal pipelines.

The service turns explicit cut ranges into validated keep spans and can inject
an intro trim span when requested. It does not perform any I/O or ffmpeg work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

CutRange = tuple[float, float]
KeepSpan = tuple[float, float | None]


@dataclass(frozen=True)
class CutPlan:
    """Validated cut ranges plus the keep spans derived from them."""

    cuts: list[CutRange]
    spans: list[KeepSpan]


def _validate_cut_ranges(cuts: Sequence[CutRange]) -> list[CutRange]:
    sorted_cuts = sorted(cuts)
    for index in range(len(sorted_cuts) - 1):
        current_cut = sorted_cuts[index]
        next_cut = sorted_cuts[index + 1]
        if current_cut[1] > next_cut[0]:
            raise ValueError(
                f"Overlapping ad segments: [{current_cut[0]}, {current_cut[1]}]"
                f" overlaps [{next_cut[0]}, {next_cut[1]}]"
            )
    return sorted_cuts


def _spans_from_cuts(cuts: Sequence[CutRange]) -> list[KeepSpan]:
    spans: list[KeepSpan] = []
    cursor = 0.0
    for start, end in cuts:
        if start > cursor:
            spans.append((cursor, start))
        cursor = end
    spans.append((cursor, None))
    return spans


def build_cut_plan(cuts: Sequence[CutRange], *, intro_trim: float | None = None) -> CutPlan:
    """Validate cuts and derive keep spans, optionally adding an intro trim.

    intro_trim: cut everything from 0.0 to this timestamp (pass intro_start to keep
    the jingle, or intro_end to remove it).
    """
    all_cuts: list[CutRange] = (
        [(0.0, intro_trim)] + list(cuts) if intro_trim is not None else list(cuts)
    )
    sorted_cuts = _validate_cut_ranges(all_cuts)
    spans = _spans_from_cuts(sorted_cuts)
    return CutPlan(cuts=sorted_cuts, spans=spans)


__all__ = ["CutPlan", "build_cut_plan"]
