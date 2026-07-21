"""Helpers for evaluating audio-match manifests against labeled clips."""

from __future__ import annotations

import json
from collections.abc import Iterable
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioManifestEvaluation:
    """Precision/recall summary for a scored match manifest."""

    manifest_path: Path
    expected_true_indices: frozenset[int]
    predicted_indices: frozenset[int]
    true_positive_indices: frozenset[int]
    false_positive_indices: frozenset[int]
    false_negative_indices: frozenset[int]
    precision: float
    recall: float
    f1: float


def _read_manifest_indices(manifest_path: Path) -> list[int]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(newline="", encoding="utf-8-sig") as manifest_file:
        reader = DictReader(manifest_file)
        if reader.fieldnames is None or "index" not in reader.fieldnames:
            raise ValueError(f"Manifest missing index column: {manifest_path}")
        return [int(row["index"]) for row in reader]


def load_match_labels(label_path: Path) -> frozenset[int]:
    """Load true-positive clip indices from a JSON label file next to a manifest."""
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    data = json.loads(label_path.read_text(encoding="utf-8"))
    indices = data.get("true_positive_indices")
    if not isinstance(indices, list):
        raise TypeError(f"Label file missing 'true_positive_indices' list: {label_path}")
    return frozenset(int(i) for i in indices)


def evaluate_match_manifest(
    *,
    manifest_path: Path,
    true_positive_indices: Iterable[int],
) -> AudioManifestEvaluation:
    """Evaluate a manifest against a labeled set of true clip indices."""
    predicted_indices = frozenset(_read_manifest_indices(manifest_path))
    expected_true_indices = frozenset(true_positive_indices)

    true_positive = predicted_indices & expected_true_indices
    false_positive = predicted_indices - expected_true_indices
    false_negative = expected_true_indices - predicted_indices

    precision = len(true_positive) / len(predicted_indices) if predicted_indices else 0.0
    recall = len(true_positive) / len(expected_true_indices) if expected_true_indices else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision > 0.0 and recall > 0.0 else 0.0

    return AudioManifestEvaluation(
        manifest_path=manifest_path,
        expected_true_indices=expected_true_indices,
        predicted_indices=predicted_indices,
        true_positive_indices=true_positive,
        false_positive_indices=false_positive,
        false_negative_indices=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


__all__ = ["AudioManifestEvaluation", "evaluate_match_manifest", "load_match_labels"]
