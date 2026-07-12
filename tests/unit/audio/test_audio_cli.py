"""Tests for audio CLI entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from part_io.adapters.audio.matcher import AudioMatch, BestMatch
from part_io.cli import audio_locate, audio_review, audio_search
from part_io.cli.lint.registry import build_tool_cmd


def test_audio_search_main_prints_matches(monkeypatch, capsys, tmp_path: Path) -> None:
    """The search CLI should print detected match windows."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_search,
        "find_audio_sample_matches",
        lambda **_kwargs: [
            AudioMatch(start_seconds=1.23, end_seconds=4.56, duration_seconds=3.33, score=0.91)
        ],
    )
    monkeypatch.setattr(sys, "argv", ["audio_search", str(source), str(sample)])

    audio_search.main()

    output = capsys.readouterr().out
    assert "1.230s -> 4.560s" in output
    assert "score=0.9100" in output


def test_audio_review_main_writes_bundle(monkeypatch, capsys, tmp_path: Path) -> None:
    """The review CLI should generate a bundle, manifest, and labels template."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_review,
        "find_audio_sample_matches",
        lambda **_kwargs: [
            AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9),
            AudioMatch(start_seconds=7.0, end_seconds=10.0, duration_seconds=3.0, score=0.8),
        ],
    )
    monkeypatch.setattr(audio_review, "_extract_clip", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audio_review",
            str(source),
            str(sample),
            "--output-root",
            str(tmp_path / "review"),
            "--bundle-name",
            "bundle",
            "--max-clips",
            "1",
        ],
    )

    audio_review.main()

    bundle_dir = tmp_path / "review" / "bundle"
    manifest_path = bundle_dir / "matches_manifest.csv"
    labels_path = bundle_dir / "match_labels.json"
    output = capsys.readouterr().out

    assert bundle_dir.exists()
    assert manifest_path.exists()
    assert labels_path.exists()
    assert "Exported clips: 1 (from 2 total matches)" in output


def test_audio_locate_main_prints_best_match(monkeypatch, capsys, tmp_path: Path) -> None:
    """The locate CLI should print the best match with its prominence."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_locate,
        "find_best_sample_match",
        lambda **_kwargs: BestMatch(
            start_seconds=6.7, end_seconds=24.7, duration_seconds=18.0, score=0.9956, prominence=4.2
        ),
    )
    monkeypatch.setattr(sys, "argv", ["audio_locate", str(source), str(sample)])

    audio_locate.main()

    output = capsys.readouterr().out
    assert "6.700s -> 24.700s" in output
    assert "prominence=4.20" in output


def test_audio_locate_main_rejects_low_prominence(monkeypatch, capsys, tmp_path: Path) -> None:
    """A peak below --min-prominence should exit non-zero with no match."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_locate,
        "find_best_sample_match",
        lambda **_kwargs: BestMatch(
            start_seconds=1.0, end_seconds=19.0, duration_seconds=18.0, score=0.98, prominence=0.5
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["audio_locate", str(source), str(sample), "--min-prominence", "3.0"]
    )

    with pytest.raises(SystemExit) as excinfo:
        audio_locate.main()

    assert excinfo.value.code == 1
    assert "No confident match found." in capsys.readouterr().out


def test_coverage_adapter_build_cmd_uses_current_python() -> None:
    """Coverage should run pytest through the current interpreter."""
    cfg = {"floor": 90}
    cmd = build_tool_cmd("coverage", cfg)

    assert cmd[:3] == ["python", "-m", "pytest"]
    assert "--cov-fail-under=90" in cmd
