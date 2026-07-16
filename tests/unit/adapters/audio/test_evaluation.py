"""Tests for the adapters.audio.evaluation module."""

from __future__ import annotations

from partio.adapters.audio.evaluation import (
    AudioManifestEvaluation,
    evaluate_match_manifest,
    load_match_labels,
)


def test_audio_manifest_evaluation_imports() -> None:
    """Verify AudioManifestEvaluation is importable."""
    assert AudioManifestEvaluation is not None


def test_evaluate_match_manifest_imports() -> None:
    """Verify evaluate_match_manifest is importable."""
    assert evaluate_match_manifest is not None


def test_load_match_labels_imports() -> None:
    """Verify load_match_labels is importable."""
    assert load_match_labels is not None
