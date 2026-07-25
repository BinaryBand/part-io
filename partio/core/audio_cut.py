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


@dataclass(frozen=True)
class RemovedSpan:
    """A contiguous span of source audio to drop, in seconds."""

    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        """Length of the removed span in seconds."""
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


def _keep_from_removed(removed: list[RemovedSpan], total_seconds: float) -> list[KeepSegment]:
    """Return the complement of *removed* over ``[0, total_seconds]``."""
    segments: list[KeepSegment] = []
    cursor = 0.0
    for span in removed:
        if span.start_seconds > cursor:
            segments.append(KeepSegment(cursor, span.start_seconds))
        cursor = max(cursor, span.end_seconds)
    if cursor < total_seconds:
        segments.append(KeepSegment(cursor, total_seconds))
    return segments


def _pair_breaks(
    openings: list[tuple[float, float]],
    closings: list[tuple[float, float]],
    total_seconds: float,
) -> list[RemovedSpan]:
    """Bracket each opening jingle with the next closing jingle after it.

    Openings and closings are matched greedily in time order: each break runs
    from an opening's start to the end of the first closing that begins at or
    after that opening ends. Openings that fall inside an already-formed break
    (extra detections of the same stinger) and closings with no preceding
    opening are ignored.
    """
    sorted_openings = sorted(openings)
    sorted_closings = sorted(closings)
    breaks: list[RemovedSpan] = []
    closing_index = 0
    guard = 0.0

    for open_start, open_end in sorted_openings:
        if breaks and open_start < guard:
            continue
        while closing_index < len(sorted_closings) and sorted_closings[closing_index][0] < open_end:
            closing_index += 1
        if closing_index >= len(sorted_closings):
            break
        close_end = min(sorted_closings[closing_index][1], total_seconds)
        breaks.append(RemovedSpan(max(open_start, 0.0), close_end))
        guard = close_end
        closing_index += 1

    return breaks


def plan_multi_cut(
    *,
    openings: list[tuple[float, float]],
    closings: list[tuple[float, float]],
    total_seconds: float,
) -> tuple[list[RemovedSpan], list[KeepSegment]]:
    """Plan removal of every ad break bracketed by an opening/closing jingle pair.

    Each element of *openings* / *closings* is a ``(start_seconds, end_seconds)``
    match. Returns the removed spans (in time order) and the surviving keep
    segments. An empty removed list means no opening jingle could be paired with
    a later closing jingle, so the caller should treat it as "no result" rather
    than writing a full copy.

    Raises ``ValueError`` when the total duration is non-positive.
    """
    if total_seconds <= 0:
        raise ValueError(f"total_seconds must be positive, got {total_seconds}")
    removed = _pair_breaks(openings, closings, total_seconds)
    return removed, _keep_from_removed(removed, total_seconds)


__all__ = ["KeepSegment", "RemovedSpan", "plan_cut", "plan_multi_cut"]
