"""Tests for audio sample matching."""

from __future__ import annotations

from pathlib import Path

from part_io.adapters.audio.matcher import AudioMatch, _suppress_overlapping, find_audio_sample_matches

ROOT = Path(__file__).resolve().parents[3]


def test_find_audio_sample_matches_reports_expected_region() -> None:
    """The detector should find the known close/open sample region near 22:46."""
    source = ROOT / "downloads" / "media" / "dece9384-9892-4b4d-9c13-5298e44d67ab.mp3"
    sample = ROOT / "downloads" / "snippets" / "close.mp3"

    matches = find_audio_sample_matches(source_path=source, sample_path=sample)

    assert matches
    assert any(1365 <= match.start_seconds <= 1367 for match in matches)


def test_matches_are_sorted_and_non_overlapping_by_default() -> None:
    """Suppression should return deduplicated matches ordered by start time."""
    source = ROOT / "downloads" / "media" / "dece9384-9892-4b4d-9c13-5298e44d67ab.mp3"
    sample = ROOT / "downloads" / "snippets" / "close.mp3"

    matches = find_audio_sample_matches(source_path=source, sample_path=sample)

    starts = [match.start_seconds for match in matches]
    assert starts == sorted(starts)
    for left, right in zip(matches, matches[1:]):
        assert right.start_seconds >= left.end_seconds or left.end_seconds - right.start_seconds < (
            0.5 * min(left.duration_seconds, right.duration_seconds)
        )


def test_suppress_overlapping_keeps_best_scored_match() -> None:
    """NMS should retain the strongest match in an overlapping cluster."""
    matches = [
        AudioMatch(start_seconds=10.0, end_seconds=14.75, duration_seconds=4.75, score=0.82),
        AudioMatch(start_seconds=10.4, end_seconds=15.15, duration_seconds=4.75, score=0.97),
        AudioMatch(start_seconds=21.0, end_seconds=25.75, duration_seconds=4.75, score=0.88),
    ]

    deduped = _suppress_overlapping(matches, min_overlap=0.5)

    assert len(deduped) == 2
    assert deduped[0].start_seconds == 10.4
    assert deduped[1].start_seconds == 21.0
