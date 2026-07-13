"""Tests for the cli.audio_search module."""

from __future__ import annotations

from part_io.cli.audio_search import main


def test_audio_search_main_imports() -> None:
    """Verify audio_search.main is importable."""
    assert main is not None
