"""Pure rendering for the audition section bar.

Maps a time region onto a fixed number of terminal columns and paints what is
known about it: territory already ruled out, territory already confirmed, the
clip currently under question, and the live playhead.  Kept free of
prompt_toolkit and I/O so the layout can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    "playhead_row",
]
