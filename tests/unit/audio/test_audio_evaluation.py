"""Tests for manifest-based audio evaluation."""

from __future__ import annotations

from pathlib import Path

from part_io.adapters.audio.evaluation import evaluate_match_manifest, load_match_labels

ROOT = Path(__file__).resolve().parents[3]
_REVIEW = ROOT / "downloads" / "snippets" / "review" / "ep_dorothy_arnold_pt1"


def test_evaluate_match_manifest_scores_labeled_close_bundle() -> None:
    """The close bundle should reflect two true positives and nine false positives."""
    bundle = _REVIEW / "close_high_points"
    labels = load_match_labels(bundle / "close_high_points_labels.json")
    result = evaluate_match_manifest(
        manifest_path=bundle / "close_high_points_manifest.csv",
        true_positive_indices=labels,
    )

    assert labels == frozenset({1, 2})
    assert result.predicted_indices == frozenset(range(1, 12))
    assert result.true_positive_indices == frozenset({1, 2})
    assert result.false_positive_indices == frozenset({3, 4, 5, 6, 7, 8, 9, 10, 11})
    assert result.false_negative_indices == frozenset()
    assert result.precision == 2 / 11
    assert result.recall == 1.0
    assert result.f1 == 4 / 13


def test_evaluate_match_manifest_scores_labeled_open_bundle() -> None:
    """The open bundle should reflect four true positives and four false positives."""
    bundle = _REVIEW / "open_high_points"
    labels = load_match_labels(bundle / "open_high_points_labels.json")
    result = evaluate_match_manifest(
        manifest_path=bundle / "open_high_points_manifest.csv",
        true_positive_indices=labels,
    )

    assert labels == frozenset({1, 2, 6, 7})
    assert result.predicted_indices == frozenset({1, 2, 3, 4, 5, 6, 7, 8})
    assert result.true_positive_indices == frozenset({1, 2, 6, 7})
    assert result.false_positive_indices == frozenset({3, 4, 5, 8})
    assert result.false_negative_indices == frozenset()
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.f1 == 2 / 3
