"""Tests for the cli.audio_locate module."""

from __future__ import annotations

from part_io.cli.audio_locate import main


def test_audio_locate_main_imports() -> None:
    """Verify audio_locate.main is importable."""
    assert main is not None
