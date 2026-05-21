"""Tests for optional refine seam."""

from __future__ import annotations

from pathlib import Path

from part_io.adapters.audio import refine_plugin
from part_io.adapters.audio.matcher import AudioMatch


def _sample_matches() -> list[AudioMatch]:
    return [AudioMatch(start_seconds=1.0, end_seconds=2.0, duration_seconds=1.0, score=0.9)]


def test_apply_optional_refine_disabled_returns_baseline(monkeypatch) -> None:
    monkeypatch.delenv("PART_IO_ENABLE_REFINE_PLUGIN", raising=False)

    matches = _sample_matches()
    result = refine_plugin.apply_optional_refine(
        matches=matches,
        source_path=Path("source.mp3"),
        sample_path=Path("sample.mp3"),
    )

    assert result == matches


def test_apply_optional_refine_missing_impl_returns_baseline(monkeypatch) -> None:
    monkeypatch.setenv("PART_IO_ENABLE_REFINE_PLUGIN", "1")
    monkeypatch.setattr(refine_plugin, "_load_refine_impl", lambda: None)

    matches = _sample_matches()
    result = refine_plugin.apply_optional_refine(
        matches=matches,
        source_path=Path("source.mp3"),
        sample_path=Path("sample.mp3"),
    )

    assert result == matches


def test_apply_optional_refine_enabled_calls_impl(monkeypatch) -> None:
    monkeypatch.setenv("PART_IO_ENABLE_REFINE_PLUGIN", "1")

    called = {"value": False}

    def _fake_refine_impl(**_kwargs):
        called["value"] = True
        return _sample_matches()

    monkeypatch.setattr(refine_plugin, "_load_refine_impl", lambda: _fake_refine_impl)
    refine_plugin.apply_optional_refine(
        matches=_sample_matches(),
        source_path=Path("source.mp3"),
        sample_path=Path("sample.mp3"),
    )
    assert called["value"]
