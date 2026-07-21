"""Pure rendering and navigation geometry for the audition section bar.

Maps a time region onto a fixed number of terminal columns and paints what is
known about it: territory already ruled out, territory already confirmed, the
clip currently under question, and the live playhead.  The same span geometry
also drives segment-to-segment jumping and the playback window, so both live
here.  Kept free of prompt_toolkit and I/O so the layout can be unit-tested
directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

GLYPH_BASE = "─"  # unexplored region
GLYPH_NO = "░"  # answered "no"
GLYPH_YES = "▒"  # answered "yes"
GLYPH_CLIP = "█"  # the clip under question
GLYPH_PLAYHEAD = "▲"

_STYLE_FOR_GLYPH = {
    GLYPH_BASE: "class:bar.base",
    GLYPH_NO: "class:bar.no",
    GLYPH_YES: "class:bar.yes",
    GLYPH_CLIP: "class:bar.clip",
}

_MIN_SPAN_SECONDS = 1e-6

# Stops closer together than this collapse into one.  Bisection answers a dozen
# overlapping probes only fractions of a second apart, and stepping through each
# of them separately would make a jump feel like it did nothing.
_STOP_TOLERANCE_SECONDS = 0.25


@dataclass(frozen=True)
class AnsweredSpan:
    """A segment the listener has already judged, and how they judged it."""

    start: float
    end: float
    answer: bool


def format_timestamp(seconds: float) -> str:
    """Render *seconds* as ``M:SS.s``.

    >>> format_timestamp(32.44)
    '0:32.4'
    >>> format_timestamp(125.0)
    '2:05.0'
    """
    minutes, remainder = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}:{remainder:04.1f}"


def _column(seconds: float, *, region_start: float, region_end: float, width: int) -> int:
    """Map a timestamp onto a bar column, clamped to the bar's bounds."""
    span = max(region_end - region_start, _MIN_SPAN_SECONDS)
    fraction = (seconds - region_start) / span
    return max(0, min(width - 1, int(fraction * width)))


def bar_cells(
    *,
    region_start: float,
    region_end: float,
    clip_start: float,
    clip_end: float,
    answered: Sequence[AnsweredSpan] = (),
    width: int,
) -> list[str]:
    """Paint the bar as *width* glyph cells.

    Answered spans are laid down first and the clip under question is painted
    over them, so the current question always stays visible.
    """
    cells = [GLYPH_BASE] * width

    def _paint(start: float, end: float, glyph: str) -> None:
        first = _column(start, region_start=region_start, region_end=region_end, width=width)
        last = _column(end, region_start=region_start, region_end=region_end, width=width)
        for column in range(first, max(first + 1, last)):
            cells[column] = glyph

    for span in answered:
        _paint(span.start, span.end, GLYPH_YES if span.answer else GLYPH_NO)
    _paint(clip_start, clip_end, GLYPH_CLIP)
    return cells


def bar_fragments(cells: Sequence[str]) -> list[tuple[str, str]]:
    """Collapse glyph cells into styled ``(style, text)`` runs."""
    return [(_STYLE_FOR_GLYPH[glyph], glyph * len(list(group))) for glyph, group in groupby(cells)]


def playhead_row(*, position: float, region_start: float, region_end: float, width: int) -> str:
    """Render the playhead marker row beneath the bar."""
    column = _column(position, region_start=region_start, region_end=region_end, width=width)
    return " " * column + GLYPH_PLAYHEAD


def segment_stops(*, clip_start: float, answered: Sequence[AnsweredSpan] = ()) -> list[float]:
    """Every point a segment jump may land on, sorted and deduped.

    A stop is the *start* of a judged span or of the clip under question -- the
    listener jumps somewhere to hear it from the beginning.  Unexplored stretches
    are not stops.

    >>> segment_stops(clip_start=30.0, answered=[AnsweredSpan(0.0, 10.0, False)])
    [0.0, 30.0]
    """
    stops: list[float] = []
    for candidate in sorted([span.start for span in answered] + [clip_start]):
        if not stops or candidate - stops[-1] > _STOP_TOLERANCE_SECONDS:
            stops.append(candidate)
    return stops


def jump_target(*, position: float, stops: Sequence[float], forward: bool) -> float | None:
    """The next stop after (or previous stop before) *position*.

    Returns ``None`` when there is nothing further in that direction, which makes
    the key a silent no-op at either end of the region rather than a jump that
    lands back where it started.

    >>> jump_target(position=0.0, stops=[0.0, 30.0], forward=True)
    30.0
    >>> jump_target(position=0.0, stops=[0.0, 30.0], forward=False) is None
    True
    """
    if forward:
        later = [stop for stop in stops if stop - position > _STOP_TOLERANCE_SECONDS]
        return later[0] if later else None
    earlier = [stop for stop in stops if position - stop > _STOP_TOLERANCE_SECONDS]
    return earlier[-1] if earlier else None


def play_window_end(
    *,
    position: float,
    clip_start: float,
    clip_end: float,
    region_end: float,
    answered: Sequence[AnsweredSpan] = (),
) -> float:
    """Where playback started at *position* should stop.

    Inside a segment, playback runs to that segment's end.  In the unexplored gap
    between segments it runs *through* to the end of the next one, so scrubbing
    back off the clip still plays the lead-in and then the clip itself instead of
    stopping at the boundary between them.
    """
    segments = [(span.start, span.end) for span in answered] + [(clip_start, clip_end)]
    # Answered spans overlap heavily during bisection, so a position can sit
    # inside several at once; the furthest end is the one worth hearing out.
    containing = [end for start, end in segments if start <= position < end]
    if containing:
        return max(containing)
    ahead = [(start, end) for start, end in segments if start > position]
    if not ahead:
        return region_end
    # The *nearest* segment ahead, not the soonest-ending one: a short probe span
    # nested inside the clip would otherwise cut playback off mid-clip.
    nearest_start = min(start for start, _end in ahead)
    return max(end for start, end in ahead if start == nearest_start)


__all__ = [
    "GLYPH_BASE",
    "GLYPH_CLIP",
    "GLYPH_NO",
    "GLYPH_PLAYHEAD",
    "GLYPH_YES",
    "AnsweredSpan",
    "bar_cells",
    "bar_fragments",
    "format_timestamp",
    "jump_target",
    "play_window_end",
    "playhead_row",
    "segment_stops",
]
