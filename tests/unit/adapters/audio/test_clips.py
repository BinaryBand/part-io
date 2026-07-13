"""Tests for the adapters.audio.clips module."""

from __future__ import annotations

from part_io.adapters.audio.clips import extract_audio_clip, play_audio_segment


def test_extract_audio_clip_imports() -> None:
    """Verify extract_audio_clip is importable."""
    assert extract_audio_clip is not None


def test_play_audio_segment_imports() -> None:
    """Verify play_audio_segment is importable."""
    assert play_audio_segment is not None
