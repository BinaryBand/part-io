"""Tests for the cli.commands.audio.bootstrap module."""

from __future__ import annotations

from partio.cli.commands.audio.bootstrap import bootstrap


def test_audio_bootstrap_command_imports() -> None:
    """Verify audio bootstrap command is importable."""
    assert bootstrap is not None
