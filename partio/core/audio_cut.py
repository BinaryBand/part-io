"""Pure planning for cutting an ad break bounded by two jingles.

Given the located spans of an opening and a closing jingle inside an episode,
compute the surviving segments to keep once the whole bracketed span -- both
jingles included -- is removed. This is deliberately I/O-free: callers supply
the already-located spans and the total duration, and the ffmpeg extraction
lives in the audio adapter. Keeping the arithmetic here makes the ordering and
bounds rules unit-testable without touching audio.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeepSegment:
    """A contiguous span of source audio to retain, in seconds."""

    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        """Length of the retained span in seconds."""
        return self.end_seconds - self.start_seconds


def _validate_span(label: str, start: float, end: float, total_seconds: float) -> None:
    if start < 0:
        raise ValueError(f"{label} start must be non-negative, got {start}")
    if end < start:
        raise ValueError(f"{label} end {end} precedes its start {start}")
    if end > total_seconds:
        raise ValueError(f"{label} end {end} exceeds total duration {total_seconds}")


def plan_cut(
    *,
    opening_start: float,
    opening_end: float,
    closing_start: float,
    closing_end: float,
    total_seconds: float,
) -> list[KeepSegment]:
    """Return the segments to keep after removing ``[opening_start, closing_end]``.

    The removed span runs from the start of the opening jingle to the end of the
    closing jingle (both jingles included), leaving the audio before the opening
    jingle and after the closing jingle. Zero-length survivors are dropped, so an
    ad break flush against the start or end of the episode yields a single kept
    segment.

    Raises ``ValueError`` when the total duration is non-positive, either span is
    out of bounds, or the opening jingle does not begin before the closing one.
    """
    if total_seconds <= 0:
        raise ValueError(f"total_seconds must be positive, got {total_seconds}")
    _validate_span("opening", opening_start, opening_end, total_seconds)
    _validate_span("closing", closing_start, closing_end, total_seconds)
    if opening_start >= closing_start:
        raise ValueError(
            f"opening jingle ({opening_start}s) must begin before the closing "
            f"jingle ({closing_start}s)"
        )

    candidates = [
        KeepSegment(0.0, opening_start),
        KeepSegment(closing_end, total_seconds),
    ]
    return [segment for segment in candidates if segment.duration_seconds > 0]


__all__ = ["KeepSegment", "plan_cut"]
