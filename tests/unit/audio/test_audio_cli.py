"""Tests for audio CLI entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

from part_io.adapters.audio.matcher import AudioMatch
from part_io.cli import audio_review, audio_search
from part_io.cli.lint import coverage


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
            AudioMatch(start_seconds=7.0, end_seconds=10.0, duration_seconds=3.0, score=0.8),
            AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9),
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
    manifest_rows = manifest_path.read_text(encoding="utf-8")

    assert bundle_dir.exists()
    assert manifest_path.exists()
    assert labels_path.exists()
    assert "Exported clips: 1 (from 2 total matches)" in output
    assert "1,0.9,2.0,5.0,3.0" in manifest_rows


def test_audio_review_main_applies_refinement(monkeypatch, capsys, tmp_path: Path) -> None:
    """The review CLI with --refine should refine coarse matches."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    # Coarse matches at 2.0 and 7.0
    coarse_matches = [
        AudioMatch(start_seconds=7.0, end_seconds=10.0, duration_seconds=3.0, score=0.8),
        AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9),
    ]

    # When refine is called, return finer matches shifted slightly from coarse
    def mock_find_matches(**kwargs):
        # If called with default step (0.1), return coarse matches
        if kwargs.get("step_seconds", 0.1) == 0.1:
            return coarse_matches
        # If called with refine step (0.01), return refined matches
        if kwargs.get("step_seconds") == 0.01:
            # Return higher-scored match offset slightly from coarse
            return [
                AudioMatch(start_seconds=2.05, end_seconds=5.05, duration_seconds=3.0, score=0.95),
                AudioMatch(start_seconds=7.05, end_seconds=10.05, duration_seconds=3.0, score=0.85),
            ]
        return coarse_matches

    monkeypatch.setattr(audio_review, "find_audio_sample_matches", mock_find_matches)
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
            "--refine",
        ],
    )

    audio_review.main()

    bundle_dir = tmp_path / "review" / "bundle"
    manifest_path = bundle_dir / "matches_manifest.csv"
    manifest_rows = manifest_path.read_text(encoding="utf-8")

    assert bundle_dir.exists()
    # With refinement, the best match should be the refined one at 2.05 with score 0.95
    assert "1,0.95,2.05" in manifest_rows or "1,0.95,2.1" in manifest_rows  # Allow minor rounding


def test_refine_match_returns_improved_match(monkeypatch, tmp_path: Path) -> None:
    """_refine_match should return best match from finer search."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    coarse_match = AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9)

    def mock_find_matches(**kwargs):
        # Return refined matches at 0.01 step around coarse location
        if kwargs.get("step_seconds") == 0.01:
            return [
                AudioMatch(start_seconds=1.95, end_seconds=4.95, duration_seconds=3.0, score=0.88),
                AudioMatch(start_seconds=2.05, end_seconds=5.05, duration_seconds=3.0, score=0.95),
                AudioMatch(start_seconds=2.10, end_seconds=5.10, duration_seconds=3.0, score=0.92),
            ]
        return [coarse_match]

    monkeypatch.setattr(audio_review, "find_audio_sample_matches", mock_find_matches)

    refined = audio_review._refine_match(
        coarse_match=coarse_match,
        source_path=source,
        sample_path=sample,
        threshold=0.8,
    )

    # Refined match should have better score than coarse
    assert refined.score > coarse_match.score
    assert refined.start_seconds == 2.05


def test_coverage_adapter_build_cmd_uses_current_python() -> None:
    """Coverage should run pytest through the current interpreter."""
    cfg = {"floor": 90}
    cmd = coverage._build_cmd(cfg)

    assert cmd[:3] == ["python", "-m", "pytest"]
    assert "--cov-fail-under=90" in cmd
