"""Tests for the cli.commands.audio.locate module."""

from __future__ import annotations

from partio.cli.commands.audio.locate import locate


def test_audio_locate_command_imports() -> None:
    """Verify audio locate command is importable."""
    assert locate is not None
