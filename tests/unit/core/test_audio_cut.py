"""Tests for the core.audio_cut planning logic."""

from __future__ import annotations

import pytest

from partio.core.audio_cut import KeepSegment, RemovedSpan, plan_cut, plan_multi_cut


def test_plan_cut_keeps_before_and_after() -> None:
    """A mid-episode break yields the audio before and after it."""
    segments = plan_cut(
        opening_start=100.0,
        opening_end=105.0,
        closing_start=160.0,
        closing_end=166.0,
        total_seconds=600.0,
    )

    assert segments == [
        KeepSegment(0.0, 100.0),
        KeepSegment(166.0, 600.0),
    ]


def test_plan_cut_removes_span_including_both_jingles() -> None:
    """The removed span spans opening_start to closing_end inclusive."""
    segments = plan_cut(
        opening_start=10.0,
        opening_end=12.0,
        closing_start=30.0,
        closing_end=33.0,
        total_seconds=100.0,
    )

    # Nothing between opening_start (10) and closing_end (33) survives.
    assert all(seg.end_seconds <= 10.0 or seg.start_seconds >= 33.0 for seg in segments)


def test_plan_cut_drops_zero_length_leading_segment() -> None:
    """A break flush against the start keeps only the tail."""
    segments = plan_cut(
        opening_start=0.0,
        opening_end=5.0,
        closing_start=20.0,
        closing_end=25.0,
        total_seconds=100.0,
    )

    assert segments == [KeepSegment(25.0, 100.0)]


def test_plan_cut_drops_zero_length_trailing_segment() -> None:
    """A break flush against the end keeps only the head."""
    segments = plan_cut(
        opening_start=80.0,
        opening_end=85.0,
        closing_start=95.0,
        closing_end=100.0,
        total_seconds=100.0,
    )

    assert segments == [KeepSegment(0.0, 80.0)]


def test_keep_segment_duration() -> None:
    """duration_seconds reflects the span length."""
    assert KeepSegment(10.0, 25.0).duration_seconds == 15.0


def test_plan_cut_rejects_non_positive_total() -> None:
    """A non-positive total duration is rejected."""
    with pytest.raises(ValueError, match="total_seconds must be positive"):
        plan_cut(
            opening_start=1.0,
            opening_end=2.0,
            closing_start=3.0,
            closing_end=4.0,
            total_seconds=0.0,
        )


def test_plan_cut_rejects_opening_after_closing() -> None:
    """The opening jingle must begin before the closing one."""
    with pytest.raises(ValueError, match="must begin before"):
        plan_cut(
            opening_start=50.0,
            opening_end=55.0,
            closing_start=50.0,
            closing_end=60.0,
            total_seconds=100.0,
        )


def test_plan_cut_rejects_out_of_bounds_span() -> None:
    """A closing span beyond the total duration is rejected."""
    with pytest.raises(ValueError, match="exceeds total duration"):
        plan_cut(
            opening_start=10.0,
            opening_end=15.0,
            closing_start=90.0,
            closing_end=120.0,
            total_seconds=100.0,
        )


def test_plan_cut_rejects_negative_start() -> None:
    """A negative span start is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        plan_cut(
            opening_start=-1.0,
            opening_end=5.0,
            closing_start=20.0,
            closing_end=25.0,
            total_seconds=100.0,
        )


def test_removed_span_duration() -> None:
    """duration_seconds reflects the span length."""
    assert RemovedSpan(100.0, 166.0).duration_seconds == 66.0


def test_plan_multi_cut_pairs_two_breaks() -> None:
    """Two opening/closing pairs yield two removed spans and three keeps."""
    removed, segments = plan_multi_cut(
        openings=[(100.0, 105.0), (300.0, 305.0)],
        closings=[(160.0, 166.0), (360.0, 366.0)],
        total_seconds=600.0,
    )

    assert removed == [RemovedSpan(100.0, 166.0), RemovedSpan(300.0, 366.0)]
    assert segments == [
        KeepSegment(0.0, 100.0),
        KeepSegment(166.0, 300.0),
        KeepSegment(366.0, 600.0),
    ]


def test_plan_multi_cut_ignores_extra_opening_inside_break() -> None:
    """A second opening detected inside a break does not start a new break."""
    removed, _segments = plan_multi_cut(
        openings=[(100.0, 105.0), (120.0, 125.0)],
        closings=[(160.0, 166.0)],
        total_seconds=600.0,
    )

    assert removed == [RemovedSpan(100.0, 166.0)]


def test_plan_multi_cut_ignores_unmatched_closings() -> None:
    """A closing with no preceding opening is ignored."""
    removed, _segments = plan_multi_cut(
        openings=[(200.0, 205.0)],
        closings=[(50.0, 56.0), (260.0, 266.0)],
        total_seconds=600.0,
    )

    assert removed == [RemovedSpan(200.0, 266.0)]


def test_plan_multi_cut_dangling_opening_is_not_a_break() -> None:
    """An opening with no later closing produces no break."""
    removed, segments = plan_multi_cut(
        openings=[(100.0, 105.0), (500.0, 505.0)],
        closings=[(160.0, 166.0)],
        total_seconds=600.0,
    )

    assert removed == [RemovedSpan(100.0, 166.0)]
    assert segments == [KeepSegment(0.0, 100.0), KeepSegment(166.0, 600.0)]


def test_plan_multi_cut_no_pairs_keeps_everything() -> None:
    """With no pairable jingles, nothing is removed and the whole file is kept."""
    removed, segments = plan_multi_cut(
        openings=[],
        closings=[(10.0, 16.0)],
        total_seconds=600.0,
    )

    assert removed == []
    assert segments == [KeepSegment(0.0, 600.0)]


def test_plan_multi_cut_clamps_closing_past_end() -> None:
    """A closing that runs past the total duration is clamped."""
    removed, segments = plan_multi_cut(
        openings=[(580.0, 585.0)],
        closings=[(595.0, 610.0)],
        total_seconds=600.0,
    )

    assert removed == [RemovedSpan(580.0, 600.0)]
    assert segments == [KeepSegment(0.0, 580.0)]


def test_plan_multi_cut_rejects_non_positive_total() -> None:
    """A non-positive total duration is rejected."""
    with pytest.raises(ValueError, match="total_seconds must be positive"):
        plan_multi_cut(openings=[(1.0, 2.0)], closings=[(3.0, 4.0)], total_seconds=0.0)
