"""Tests for the cli.library._cache module."""

from __future__ import annotations

from pathlib import Path

from partio.cli.library import _cache
from partio.core.ports import AudioPathKind


def test_remember_indexes_a_path() -> None:
    """A remembered path is retrievable with its label and kind intact."""
    _cache.remember(Path("a.mp3"), label="Ep A", kind=AudioPathKind.SOURCE)

    (entry,) = _cache.cached()
    assert (entry.label, entry.kind) == ("Ep A", AudioPathKind.SOURCE)


def test_remember_ignores_an_already_indexed_path() -> None:
    """Re-remembering the same path does not create a duplicate entry."""
    _cache.remember(Path("a.mp3"), label="Ep A", kind=AudioPathKind.SOURCE)
    _cache.remember(Path("a.mp3"), label="Ep A again", kind=AudioPathKind.SOURCE)

    assert len(_cache.cached()) == 1


def test_cached_filters_by_kind() -> None:
    """Asking for one kind never returns the other."""
    _cache.remember(Path("ep.mp3"), label="Ep", kind=AudioPathKind.SOURCE)
    _cache.remember(Path("seed.mp3"), label="Seed", kind=AudioPathKind.SAMPLE)

    assert [e.label for e in _cache.cached(AudioPathKind.SAMPLE)] == ["Seed"]
    assert [e.label for e in _cache.cached(AudioPathKind.SOURCE)] == ["Ep"]


def test_cached_is_empty_when_the_index_is_missing() -> None:
    """No index yet reads as an empty library."""
    assert _cache.cached() == []


def test_cached_tolerates_a_corrupt_index(monkeypatch, tmp_path) -> None:
    """A broken index costs a re-download, not a working prompt."""
    broken = tmp_path / "library.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(_cache, "DEFAULT_LIBRARY_PATH", broken)

    assert _cache.cached() == []
