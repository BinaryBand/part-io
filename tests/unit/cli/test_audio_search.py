"""Tests for the cli.commands.audio.search module."""

from __future__ import annotations

from partio.cli.commands.audio.search import search


def test_audio_search_command_imports() -> None:
    """Verify audio search command is importable."""
    assert search is not None
