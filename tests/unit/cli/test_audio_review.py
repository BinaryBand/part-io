"""Tests for the cli.audio_review module."""

from __future__ import annotations

from part_io.cli.audio_review import build_interactive_auditor, review


def test_audio_review_command_imports() -> None:
    """Verify audio_review.review is importable."""
    assert review is not None


def test_build_interactive_auditor_imports() -> None:
    """Verify build_interactive_auditor is importable."""
    assert build_interactive_auditor is not None
