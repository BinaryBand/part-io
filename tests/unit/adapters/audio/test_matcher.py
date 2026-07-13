"""Tests for the adapters.audio.matcher module."""

from __future__ import annotations

from part_io.adapters.audio.matcher import AudioMatch, find_audio_sample_matches


def test_audio_match_imports() -> None:
    """Verify AudioMatch is importable."""
    assert AudioMatch is not None


def test_find_audio_sample_matches_imports() -> None:
    """Verify find_audio_sample_matches is importable."""
    assert find_audio_sample_matches is not None
