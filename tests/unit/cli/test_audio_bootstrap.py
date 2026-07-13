"""Tests for the cli.audio_bootstrap module."""

from __future__ import annotations

from part_io.cli.audio_bootstrap import bootstrap


def test_audio_bootstrap_command_imports() -> None:
    """Verify audio_bootstrap.bootstrap is importable."""
    assert bootstrap is not None
