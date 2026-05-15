"""Tests for manifest-based audio evaluation."""

from __future__ import annotations

import json
from csv import DictWriter
from pathlib import Path

from part_io.adapters.audio.evaluation import evaluate_match_manifest, load_match_labels


def _write_manifest(manifest_path: Path, indices: list[int]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = DictWriter(manifest_file, fieldnames=["index", "score"]) 
        writer.writeheader()
        for index in indices:
            writer.writerow({"index": index, "score": 0.9})


def _write_labels(label_path: Path, true_indices: list[int]) -> None:
    label_path.write_text(
        json.dumps({"true_positive_indices": true_indices}, indent=2) + "\n",
        encoding="utf-8",
    )


def test_evaluate_match_manifest_scores_expected_sets(tmp_path: Path) -> None:
    """Evaluator should compute TP/FP/FN and precision/recall/f1 from label files."""
    manifest_path = tmp_path / "matches_manifest.csv"
    labels_path = tmp_path / "match_labels.json"

    _write_manifest(manifest_path, indices=[1, 2, 3, 4, 5])
    _write_labels(labels_path, true_indices=[2, 5, 6])

    labels = load_match_labels(labels_path)
    result = evaluate_match_manifest(
        manifest_path=manifest_path,
        true_positive_indices=labels,
    )

    assert labels == frozenset({2, 5, 6})
    assert result.predicted_indices == frozenset({1, 2, 3, 4, 5})
    assert result.true_positive_indices == frozenset({2, 5})
    assert result.false_positive_indices == frozenset({1, 3, 4})
    assert result.false_negative_indices == frozenset({6})
    assert result.precision == 2 / 5
    assert result.recall == 2 / 3
    assert result.f1 == 4 / 7


def test_evaluate_match_manifest_handles_empty_predictions(tmp_path: Path) -> None:
    """Evaluator should return zero precision/recall/f1 when there are no predictions."""
    manifest_path = tmp_path / "matches_manifest.csv"
    labels_path = tmp_path / "match_labels.json"

    _write_manifest(manifest_path, indices=[])
    _write_labels(labels_path, true_indices=[1, 3])

    labels = load_match_labels(labels_path)
    result = evaluate_match_manifest(
        manifest_path=manifest_path,
        true_positive_indices=labels,
    )

    assert labels == frozenset({1, 3})
    assert result.predicted_indices == frozenset()
    assert result.true_positive_indices == frozenset()
    assert result.false_positive_indices == frozenset()
    assert result.false_negative_indices == frozenset({1, 3})
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
