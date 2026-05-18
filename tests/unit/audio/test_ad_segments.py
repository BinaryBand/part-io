"""Unit tests for ad_segments adapter and audio_ad_remove filter builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from part_io.adapters.audio.ad_segments import (
    AdSegment,
    load_manifest_matches,
    pair_ad_segments,
)
from part_io.adapters.audio.matcher import AudioMatch
from part_io.cli.audio_ad_remove import (
    _build_filter_complex,
    _build_keep_spans,
    _validate_segments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "score",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "clip_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_labels(path: Path, true_positive_indices: list[int]) -> None:
    path.write_text(
        json.dumps({"true_positive_indices": true_positive_indices, "false_positive_indices": []}),
        encoding="utf-8",
    )


def _match(start: float, duration: float = 10.0, score: float = 0.9) -> AudioMatch:
    return AudioMatch(
        start_seconds=start,
        end_seconds=start + duration,
        duration_seconds=duration,
        score=score,
    )


# ---------------------------------------------------------------------------
# load_manifest_matches
# ---------------------------------------------------------------------------


def test_load_manifest_matches_returns_all_rows_without_labels(tmp_path: Path) -> None:
    manifest = tmp_path / "matches_manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "index": 1,
                "score": 0.9,
                "start_seconds": 100.0,
                "end_seconds": 110.0,
                "duration_seconds": 10.0,
                "clip_path": "",
            },
            {
                "index": 2,
                "score": 0.8,
                "start_seconds": 200.0,
                "end_seconds": 210.0,
                "duration_seconds": 10.0,
                "clip_path": "",
            },
        ],
    )

    matches = load_manifest_matches(manifest)

    assert len(matches) == 2
    assert matches[0].start_seconds == 100.0
    assert matches[1].start_seconds == 200.0


def test_load_manifest_matches_filters_to_true_positives(tmp_path: Path) -> None:
    manifest = tmp_path / "matches_manifest.csv"
    labels = tmp_path / "match_labels.json"
    _write_manifest(
        manifest,
        [
            {
                "index": 1,
                "score": 0.9,
                "start_seconds": 100.0,
                "end_seconds": 110.0,
                "duration_seconds": 10.0,
                "clip_path": "",
            },
            {
                "index": 2,
                "score": 0.8,
                "start_seconds": 200.0,
                "end_seconds": 210.0,
                "duration_seconds": 10.0,
                "clip_path": "",
            },
            {
                "index": 3,
                "score": 0.85,
                "start_seconds": 300.0,
                "end_seconds": 310.0,
                "duration_seconds": 10.0,
                "clip_path": "",
            },
        ],
    )
    _write_labels(labels, true_positive_indices=[1, 3])

    matches = load_manifest_matches(manifest, labels)

    assert len(matches) == 2
    assert {m.start_seconds for m in matches} == {100.0, 300.0}


def test_load_manifest_matches_falls_back_when_labels_empty(tmp_path: Path) -> None:
    manifest = tmp_path / "matches_manifest.csv"
    labels = tmp_path / "match_labels.json"
    _write_manifest(
        manifest,
        [
            {
                "index": 1,
                "score": 0.9,
                "start_seconds": 100.0,
                "end_seconds": 110.0,
                "duration_seconds": 10.0,
                "clip_path": "",
            },
        ],
    )
    _write_labels(labels, true_positive_indices=[])

    matches = load_manifest_matches(manifest, labels)

    assert len(matches) == 1


def test_load_manifest_matches_raises_on_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest_matches(tmp_path / "missing.csv")


# ---------------------------------------------------------------------------
# pair_ad_segments
# ---------------------------------------------------------------------------


def test_pair_ad_segments_basic_pairing() -> None:
    opens = [_match(100.0)]
    closes = [_match(200.0)]

    segments, unpaired_opens, unpaired_closes = pair_ad_segments(opens, closes)

    assert len(segments) == 1
    assert segments[0].open_start == 100.0
    assert segments[0].close_end == 210.0
    assert unpaired_opens == []
    assert unpaired_closes == []


def test_pair_ad_segments_multiple_pairs() -> None:
    opens = [_match(100.0), _match(500.0)]
    closes = [_match(300.0), _match(700.0)]

    segments, unpaired_opens, unpaired_closes = pair_ad_segments(opens, closes)

    assert len(segments) == 2
    assert segments[0].open_start == 100.0
    assert segments[1].open_start == 500.0
    assert unpaired_opens == []
    assert unpaired_closes == []


def test_pair_ad_segments_close_too_soon_is_skipped() -> None:
    opens = [_match(100.0)]
    # Gap = 105 - 110 = -5 (close starts before open ends) → below min_gap
    closes = [_match(105.0)]

    segments, unpaired_opens, unpaired_closes = pair_ad_segments(opens, closes, min_gap=10.0)

    assert segments == []
    assert len(unpaired_opens) == 1
    assert len(unpaired_closes) == 1


def test_pair_ad_segments_close_too_far_is_skipped() -> None:
    opens = [_match(100.0)]
    closes = [_match(800.0)]  # gap = 800 - 110 = 690s > max_gap=600

    segments, unpaired_opens, unpaired_closes = pair_ad_segments(opens, closes, max_gap=600.0)

    assert segments == []
    assert len(unpaired_opens) == 1
    assert len(unpaired_closes) == 1


def test_pair_ad_segments_each_close_used_at_most_once() -> None:
    # Two opens both in range of the same close — first open wins
    opens = [_match(100.0), _match(120.0)]
    closes = [_match(200.0)]

    segments, unpaired_opens, unpaired_closes = pair_ad_segments(opens, closes)

    assert len(segments) == 1
    assert segments[0].open_start == 100.0
    assert len(unpaired_opens) == 1
    assert unpaired_opens[0].start_seconds == 120.0


def test_pair_ad_segments_returns_gap_seconds() -> None:
    opens = [_match(100.0, duration=10.0)]  # ends at 110
    closes = [_match(200.0, duration=10.0)]  # starts at 200, gap = 200 - 110 = 90

    segments, _, _ = pair_ad_segments(opens, closes)

    assert segments[0].gap_seconds == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# _build_keep_spans and _build_filter_complex
# ---------------------------------------------------------------------------


def _seg(open_start: float, close_end: float) -> AdSegment:
    return AdSegment(
        open_start=open_start,
        open_end=open_start + 10.0,
        close_start=close_end - 10.0,
        close_end=close_end,
        open_score=0.9,
        close_score=0.9,
    )


def test_build_keep_spans_single_cut() -> None:
    segments = [_seg(100.0, 200.0)]
    spans = _build_keep_spans(segments)

    assert spans == [(0.0, 100.0), (200.0, None)]


def test_build_keep_spans_multiple_cuts() -> None:
    segments = [_seg(100.0, 200.0), _seg(400.0, 500.0)]
    spans = _build_keep_spans(segments)

    assert spans == [(0.0, 100.0), (200.0, 400.0), (500.0, None)]


def test_build_keep_spans_cut_from_start() -> None:
    segments = [_seg(0.0, 100.0)]
    spans = _build_keep_spans(segments)

    # No leading span since cut starts at 0
    assert spans == [(100.0, None)]


def test_build_filter_complex_single_span() -> None:
    spans = [(0.0, 100.0), (200.0, None)]
    fc, n = _build_filter_complex(spans)

    assert n == 2
    assert "atrim=0.000:100.000" in fc
    assert "atrim=200.000" in fc
    assert "concat=n=2:v=0:a=1[out]" in fc


def test_validate_segments_raises_on_overlap() -> None:
    segments = [_seg(100.0, 250.0), _seg(200.0, 350.0)]  # overlap at 200-250

    with pytest.raises(ValueError, match="Overlapping"):
        _validate_segments(segments)


def test_validate_segments_passes_non_overlapping() -> None:
    segments = [_seg(100.0, 200.0), _seg(300.0, 400.0)]
    _validate_segments(segments)  # should not raise
