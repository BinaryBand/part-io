"""Tests for the audition section-bar renderer."""

from __future__ import annotations

from partio.cli.commands.audio._section_bar import (
    GLYPH_BASE,
    GLYPH_CLIP,
    GLYPH_NO,
    GLYPH_PLAYHEAD,
    GLYPH_YES,
    AnsweredSpan,
    bar_cells,
    bar_fragments,
    format_timestamp,
    playhead_row,
)

# -- format_timestamp --------------------------------------------------------


def test_format_timestamp_renders_minutes_and_tenths() -> None:
    """Seconds render as M:SS.s with a zero-padded seconds field."""
    assert format_timestamp(32.44) == "0:32.4"
    assert format_timestamp(125.0) == "2:05.0"
    assert format_timestamp(5.0) == "0:05.0"


def test_format_timestamp_clamps_negatives() -> None:
    """A negative position never renders a negative clock."""
    assert format_timestamp(-3.0) == "0:00.0"


# -- bar_cells ---------------------------------------------------------------


def test_bar_cells_marks_the_clip_under_question() -> None:
    """The clip occupies its proportional slice of the bar."""
    cells = bar_cells(region_start=0.0, region_end=100.0, clip_start=50.0, clip_end=60.0, width=100)

    assert len(cells) == 100
    assert set(cells[50:60]) == {GLYPH_CLIP}
    assert set(cells[:50]) == {GLYPH_BASE}
    assert set(cells[60:]) == {GLYPH_BASE}


def test_bar_cells_shades_answered_spans() -> None:
    """Answered spans render as yes/no shading distinct from unexplored region."""
    answered = [
        AnsweredSpan(start=0.0, end=10.0, answer=False),
        AnsweredSpan(start=10.0, end=20.0, answer=True),
    ]
    cells = bar_cells(
        region_start=0.0,
        region_end=100.0,
        clip_start=50.0,
        clip_end=60.0,
        answered=answered,
        width=100,
    )

    assert set(cells[0:10]) == {GLYPH_NO}
    assert set(cells[10:20]) == {GLYPH_YES}
    assert set(cells[20:50]) == {GLYPH_BASE}


def test_bar_cells_paints_clip_over_history() -> None:
    """The current question stays visible even where it overlaps answered spans."""
    answered = [AnsweredSpan(start=45.0, end=65.0, answer=False)]
    cells = bar_cells(
        region_start=0.0,
        region_end=100.0,
        clip_start=50.0,
        clip_end=60.0,
        answered=answered,
        width=100,
    )

    assert set(cells[50:60]) == {GLYPH_CLIP}
    assert set(cells[45:50]) == {GLYPH_NO}


def test_bar_cells_clamps_out_of_range_spans() -> None:
    """Spans beyond the region are clamped rather than raising."""
    cells = bar_cells(
        region_start=10.0,
        region_end=20.0,
        clip_start=-100.0,
        clip_end=999.0,
        width=40,
    )

    assert len(cells) == 40
    assert cells[0] == GLYPH_CLIP


def test_bar_cells_zero_width_region_does_not_divide_by_zero() -> None:
    """A degenerate region still renders a bar."""
    cells = bar_cells(region_start=5.0, region_end=5.0, clip_start=5.0, clip_end=5.0, width=10)

    assert len(cells) == 10


# -- bar_fragments -----------------------------------------------------------


def test_bar_fragments_collapse_runs() -> None:
    """Adjacent identical glyphs merge into a single styled run."""
    fragments = bar_fragments([GLYPH_BASE, GLYPH_BASE, GLYPH_CLIP, GLYPH_CLIP, GLYPH_CLIP])

    assert len(fragments) == 2
    assert fragments[0][1] == GLYPH_BASE * 2
    assert fragments[1][1] == GLYPH_CLIP * 3
    assert fragments[0][0] != fragments[1][0]


def test_bar_fragments_text_round_trips() -> None:
    """Joining the fragment texts reproduces the original cells."""
    cells = bar_cells(region_start=0.0, region_end=60.0, clip_start=20.0, clip_end=30.0, width=64)

    assert "".join(text for _style, text in bar_fragments(cells)) == "".join(cells)


# -- playhead_row ------------------------------------------------------------


def test_playhead_row_positions_the_marker() -> None:
    """The marker sits at the column matching the playback position."""
    row = playhead_row(position=50.0, region_start=0.0, region_end=100.0, width=100)

    assert row.index(GLYPH_PLAYHEAD) == 50


def test_playhead_row_clamps_to_the_bar() -> None:
    """A position past the region end stays inside the bar."""
    row = playhead_row(position=500.0, region_start=0.0, region_end=100.0, width=40)

    assert row.index(GLYPH_PLAYHEAD) == 39
