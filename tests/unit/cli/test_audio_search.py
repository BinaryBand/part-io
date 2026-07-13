"""Tests for the cli.audio_search module."""

from __future__ import annotations

from part_io.cli.audio_search import search


def test_audio_search_command_imports() -> None:
    """Verify audio_search.search is importable."""
    assert search is not None
