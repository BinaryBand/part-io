"""Tests for the cli.commands.audio.review module."""

from __future__ import annotations

from partio.cli.commands.audio._auditor import build_interactive_auditor
from partio.cli.commands.audio.review import review


def test_audio_review_command_imports() -> None:
    """Verify audio review command is importable."""
    assert review is not None


def test_build_interactive_auditor_imports() -> None:
    """Verify build_interactive_auditor is importable."""
    assert build_interactive_auditor is not None
