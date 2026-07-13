"""Tests for the cli.audio_locate module."""

from __future__ import annotations

from part_io.cli.audio_locate import locate


def test_audio_locate_command_imports() -> None:
    """Verify audio_locate.locate is importable."""
    assert locate is not None
