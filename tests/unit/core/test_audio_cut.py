"""Tests for the core.audio_cut planning logic."""

from __future__ import annotations

import pytest

from partio.core.audio_cut import KeepSegment, plan_cut


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
