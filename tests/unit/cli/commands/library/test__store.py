"""Tests for the cli.commands.library._store module."""

from __future__ import annotations

from pathlib import Path

from part_io.cli.commands.library import _store
from part_io.core.ports import AudioPathEntry, AudioPathKind


def test_default_store_round_trips_an_entry(tmp_path: Path, monkeypatch) -> None:
    """default_store() should serialize and deserialize an AudioPathEntry unchanged."""
    monkeypatch.setattr(_store, "DEFAULT_LIBRARY_PATH", tmp_path / "library.json")

    entry = AudioPathEntry(
        id="a", path=tmp_path / "episode.mp3", label="Episode", kind=AudioPathKind.SOURCE
    )
    _store.default_store().add_item(entry)

    assert _store.default_store().list_items() == [entry]


def test_default_store_honors_explicit_path(tmp_path: Path) -> None:
    """Passing an explicit path should bypass DEFAULT_LIBRARY_PATH entirely."""
    explicit_path = tmp_path / "explicit.json"
    entry = AudioPathEntry(
        id="b", path=tmp_path / "sample.mp3", label="Sample", kind=AudioPathKind.SAMPLE
    )

    _store.default_store(explicit_path).add_item(entry)

    assert explicit_path.exists()
    assert _store.default_store(explicit_path).list_items() == [entry]
