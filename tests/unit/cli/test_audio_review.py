"""Tests for the cli.audio_review module."""

from __future__ import annotations

from part_io.cli.audio_review import build_interactive_auditor, main


def test_audio_review_main_imports() -> None:
    """Verify audio_review.main is importable."""
    assert main is not None


def test_build_interactive_auditor_imports() -> None:
    """Verify build_interactive_auditor is importable."""
    assert build_interactive_auditor is not None
