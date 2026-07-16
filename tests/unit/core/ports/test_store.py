"""Tests for the core.ports.store module."""

from __future__ import annotations

from pathlib import Path

import pytest

from part_io.core.ports.store import AudioPathEntry, AudioPathKind


def test_audio_path_kind_values() -> None:
    """AudioPathKind should expose the two documented string values."""
    assert AudioPathKind.SOURCE.value == "source"
    assert AudioPathKind.SAMPLE.value == "sample"


def test_audio_path_entry_is_frozen() -> None:
    """AudioPathEntry instances must be immutable."""
    entry = AudioPathEntry(id="abc", path=Path("x.mp3"), label="x", kind=AudioPathKind.SOURCE)
    with pytest.raises(AttributeError):
        entry.label = "y"  # ty: ignore[invalid-assignment]
