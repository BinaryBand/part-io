"""Tests for the cli.audio_bootstrap module."""

from __future__ import annotations

from part_io.cli.audio_bootstrap import main


def test_audio_bootstrap_main_imports() -> None:
    """Verify audio_bootstrap.main is importable."""
    assert main is not None
