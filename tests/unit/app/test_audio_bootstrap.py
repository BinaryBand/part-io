"""Tests for the human-in-the-loop jingle discovery service."""

from __future__ import annotations

import pytest

from partio.app.audio_bootstrap import locate_jingle_span, locate_jingle_spans
from partio.core.ports.audio import AuditorFn  # noqa: TC001

AuditorCall = tuple[float, float, str]


def _make_multi_span_auditor(
    jingles: list[tuple[float, float]], calls: list[AuditorCall] | None = None
) -> AuditorFn:
    """Answer auditor questions from a list of known true jingle spans."""

    def _auditor(start_seconds: float, duration_seconds: float, question: str) -> bool:
        if calls is not None:
            calls.append((start_seconds, duration_seconds, question))
        clip_end = start_seconds + duration_seconds
        if "anywhere" in question:
            return any(start_seconds < end and clip_end > start for start, end in jingles)
        if "STARTS" in question:
            return any(start <= start_seconds <= end for start, end in jingles)
        if "ENDS" in question:
            return any(start <= clip_end <= end for start, end in jingles)
        raise AssertionError(f"Unexpected question: {question}")

    return _auditor


def _make_synthetic_auditor(
    jingle_start: float, jingle_end: float, calls: list[AuditorCall] | None = None
) -> AuditorFn:
    """Answer auditor questions from a known true jingle span."""
    return _make_multi_span_auditor([(jingle_start, jingle_end)], calls)


def test_locate_jingle_span_converges_within_resolution() -> None:
    """A mid-region jingle should be pinned to within the bisection resolution."""
    resolution = 0.5
    auditor = _make_synthetic_auditor(47.0, 65.0)

    span = locate_jingle_span(
        auditor=auditor, region_start=0.0, region_end=120.0, resolution=resolution
    )

    assert span is not None
    onset, offset = span
    assert 47.0 <= onset <= 47.0 + resolution
    assert 65.0 - resolution <= offset <= 65.0


def test_locate_jingle_span_returns_none_outside_region() -> None:
    """A jingle beyond the search region should yield no discovery."""
    auditor = _make_synthetic_auditor(200.0, 218.0)

    span = locate_jingle_span(auditor=auditor, region_start=0.0, region_end=120.0)

    assert span is None


def test_locate_jingle_span_overlapping_first_tile_uses_region_start() -> None:
    """With no preceding "no" tile, the onset lower bound is the region start."""
    resolution = 0.5
    auditor = _make_synthetic_auditor(3.0, 20.0)

    span = locate_jingle_span(
        auditor=auditor, region_start=0.0, region_end=120.0, resolution=resolution
    )

    assert span is not None
    onset, offset = span
    assert 3.0 <= onset <= 3.0 + resolution
    assert 20.0 - resolution <= offset <= 20.0


def test_locate_jingle_span_running_past_region_end_caps_offset() -> None:
    """A jingle running past the region end is capped at the region end."""
    resolution = 0.5
    auditor = _make_synthetic_auditor(110.0, 130.0)

    span = locate_jingle_span(
        auditor=auditor, region_start=0.0, region_end=120.0, resolution=resolution
    )

    assert span is not None
    onset, offset = span
    assert 110.0 <= onset <= 110.0 + resolution
    assert 120.0 - resolution <= offset <= 120.0


def test_locate_jingle_span_returns_yes_tile_bounds_when_no_probe_hits() -> None:
    """A sub-probe-length jingle falls back to the unrefined yes-tile bounds."""
    auditor = _make_synthetic_auditor(33.2, 33.4)

    span = locate_jingle_span(auditor=auditor, region_start=0.0, region_end=60.0, probe_seconds=1.5)

    assert span == (30.0, 40.0)


def test_locate_jingle_span_asks_monotone_tuning_predicates() -> None:
    """Tuning answers, ordered by probe time, must flip at most once."""
    calls: list[AuditorCall] = []
    auditor = _make_synthetic_auditor(47.0, 65.0, calls)

    locate_jingle_span(auditor=auditor, region_start=0.0, region_end=120.0)

    onset_answers = [
        auditor(start, duration, question)
        for start, duration, question in sorted(c for c in calls if "STARTS" in c[2])
    ]
    offset_answers = [
        auditor(start, duration, question)
        for start, duration, question in sorted(c for c in calls if "ENDS" in c[2])
    ]
    assert onset_answers == sorted(onset_answers)
    assert offset_answers == sorted(offset_answers, reverse=True)


def test_locate_jingle_span_rejects_empty_region() -> None:
    """An empty or inverted region should raise ValueError."""
    auditor = _make_synthetic_auditor(0.0, 1.0)

    with pytest.raises(ValueError, match="region_end"):
        locate_jingle_span(auditor=auditor, region_start=60.0, region_end=60.0)


def test_locate_jingle_span_rejects_non_positive_parameters() -> None:
    """Non-positive tuning parameters should raise ValueError."""
    auditor = _make_synthetic_auditor(0.0, 1.0)

    with pytest.raises(ValueError, match="positive"):
        locate_jingle_span(auditor=auditor, region_start=0.0, region_end=60.0, resolution=0.0)


def test_locate_jingle_spans_finds_all_occurrences_in_order() -> None:
    """Every non-overlapping occurrence should be found, in order, within resolution."""
    resolution = 0.5
    truths = [(47.0, 65.0), (130.0, 148.0), (200.0, 218.0)]
    auditor = _make_multi_span_auditor(truths)

    spans = locate_jingle_spans(auditor=auditor, region_start=0.0, region_end=240.0)

    assert len(spans) == len(truths)
    for (onset, offset), (true_start, true_end) in zip(spans, truths, strict=True):
        assert true_start <= onset <= true_start + resolution
        assert true_end - resolution <= offset <= true_end


def test_locate_jingle_spans_caps_at_max_occurrences() -> None:
    """The occurrence cap should stop discovery early."""
    auditor = _make_multi_span_auditor([(47.0, 65.0), (130.0, 148.0), (200.0, 218.0)])

    spans = locate_jingle_spans(
        auditor=auditor, region_start=0.0, region_end=240.0, max_occurrences=2
    )

    assert len(spans) == 2


def test_locate_jingle_spans_returns_empty_list_when_no_jingle() -> None:
    """A region with no jingle should yield an empty list, not None."""
    auditor = _make_multi_span_auditor([])

    spans = locate_jingle_spans(auditor=auditor, region_start=0.0, region_end=120.0)

    assert spans == []


def test_locate_jingle_spans_running_past_region_end_caps_offset() -> None:
    """A final jingle running past the region end is capped at the region end."""
    resolution = 0.5
    auditor = _make_multi_span_auditor([(20.0, 38.0), (110.0, 130.0)])

    spans = locate_jingle_spans(
        auditor=auditor, region_start=0.0, region_end=120.0, resolution=resolution
    )

    assert len(spans) == 2
    onset, offset = spans[1]
    assert 110.0 <= onset <= 110.0 + resolution
    assert 120.0 - resolution <= offset <= 120.0


def test_locate_jingle_spans_rejects_non_positive_max_occurrences() -> None:
    """A zero or negative occurrence cap should raise ValueError."""
    auditor = _make_multi_span_auditor([(0.0, 1.0)])

    with pytest.raises(ValueError, match="max_occurrences"):
        locate_jingle_spans(auditor=auditor, region_start=0.0, region_end=60.0, max_occurrences=0)


def test_locate_jingle_spans_rejects_empty_region() -> None:
    """An empty or inverted region should raise ValueError."""
    auditor = _make_multi_span_auditor([(0.0, 1.0)])

    with pytest.raises(ValueError, match="region_end"):
        locate_jingle_spans(auditor=auditor, region_start=60.0, region_end=60.0)


def test_locate_jingle_spans_rejects_non_positive_parameters() -> None:
    """Non-positive tuning parameters should raise ValueError."""
    auditor = _make_multi_span_auditor([(0.0, 1.0)])

    with pytest.raises(ValueError, match="positive"):
        locate_jingle_spans(auditor=auditor, region_start=0.0, region_end=60.0, resolution=0.0)
