"""Tests for audio CLI entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from part_io.adapters.audio import matcher as audio_matcher
from part_io.adapters.audio.matcher import AudioMatch
from part_io.cli import audio_review, audio_review_batch, audio_search
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


def test_coverage_adapter_build_cmd_uses_current_python() -> None:
    """Coverage should run pytest through the current interpreter."""
    cfg = {"floor": 90}
    cmd = coverage._build_cmd(cfg)

    assert cmd[:3] == [sys.executable, "-m", "pytest"]
    assert "--cov-fail-under=90" in cmd


# ---------------------------------------------------------------------------
# Phase 2 – Onset Anchoring
# ---------------------------------------------------------------------------


def test_anchor_to_onset_shifts_start(monkeypatch, tmp_path: Path) -> None:
    """anchor_to_onset should advance start_seconds to the first loud sample."""
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")

    # Simulate 1 second of silence then 1 second of loud signal at 16 kHz
    silence = [0] * 16000
    loud = [10000] * 16000
    pcm_samples = silence + loud

    monkeypatch.setattr(
        audio_matcher,
        "_decode_pcm_mono_16k_window",
        lambda *_args, **_kwargs: pcm_samples,
    )

    match = AudioMatch(start_seconds=0.0, end_seconds=2.0, duration_seconds=2.0, score=0.9)
    anchored = audio_matcher.anchor_to_onset(match=match, source_path=source)

    # Onset should be detected at or near the 1-second mark
    assert anchored.start_seconds >= 0.9
    assert anchored.duration_seconds == match.duration_seconds
    assert anchored.score == match.score


def test_anchor_to_onset_returns_original_on_empty_pcm(monkeypatch, tmp_path: Path) -> None:
    """anchor_to_onset should return the original match when PCM is empty."""
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")

    monkeypatch.setattr(
        audio_matcher,
        "_decode_pcm_mono_16k_window",
        lambda *_args, **_kwargs: [],
    )

    match = AudioMatch(start_seconds=5.0, end_seconds=8.0, duration_seconds=3.0, score=0.85)
    result = audio_matcher.anchor_to_onset(match=match, source_path=source)

    assert result == match


def test_audio_review_main_applies_onset_anchor(monkeypatch, capsys, tmp_path: Path) -> None:
    """The review CLI with --onset-anchor should call anchor_to_onset on each match."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    base_match = AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9)
    anchored_match = AudioMatch(start_seconds=2.4, end_seconds=5.4, duration_seconds=3.0, score=0.9)

    monkeypatch.setattr(
        audio_review,
        "find_audio_sample_matches",
        lambda **_kwargs: [base_match],
    )
    monkeypatch.setattr(
        audio_review,
        "anchor_to_onset",
        lambda **_kwargs: anchored_match,
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
            "--onset-anchor",
        ],
    )

    audio_review.main()

    manifest_path = tmp_path / "review" / "bundle" / "matches_manifest.csv"
    manifest_rows = manifest_path.read_text(encoding="utf-8")
    assert "2.4" in manifest_rows


# ---------------------------------------------------------------------------
# Phase 0 – Batch parallelism flag passthrough
# ---------------------------------------------------------------------------


def _make_batch_argv(tmp_path: Path, *, extra_flags: list[str] | None = None) -> list[str]:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "ep1.mp3").write_bytes(b"x")
    snippets_dir = tmp_path / "snippets"
    snippets_dir.mkdir()
    (snippets_dir / "close.mp3").write_bytes(b"x")
    (snippets_dir / "open.mp3").write_bytes(b"x")
    argv = [
        "audio_review_batch",
        "--media-dir",
        str(media_dir),
        "--snippets-dir",
        str(snippets_dir),
        "--output-root",
        str(tmp_path / "review"),
        "--workers",
        "1",
    ]
    if extra_flags:
        argv.extend(extra_flags)
    return argv


def test_batch_passes_onset_anchor_flag(monkeypatch, tmp_path: Path) -> None:
    """--onset-anchor should be forwarded to every _run_one call."""
    monkeypatch.setattr(sys, "argv", _make_batch_argv(tmp_path, extra_flags=["--onset-anchor"]))

    with patch.object(audio_review_batch, "_run_one", return_value=0) as mock_run:
        audio_review_batch.main()

    assert all(c.kwargs["onset_anchor"] is True for c in mock_run.call_args_list)
    assert mock_run.call_count == 2  # close + open for one media file


def test_batch_workers_controls_parallelism(monkeypatch, tmp_path: Path) -> None:
    """--workers value should be respected; all jobs still complete."""
    monkeypatch.setattr(sys, "argv", _make_batch_argv(tmp_path, extra_flags=["--workers", "2"]))

    with patch.object(audio_review_batch, "_run_one", return_value=0) as mock_run:
        audio_review_batch.main()

    assert mock_run.call_count == 2


def test_batch_exits_on_nonzero_return(monkeypatch, tmp_path: Path) -> None:
    """A non-zero exit from any job should propagate as SystemExit."""
    monkeypatch.setattr(sys, "argv", _make_batch_argv(tmp_path))

    with patch.object(audio_review_batch, "_run_one", return_value=1):
        try:
            audio_review_batch.main()
            raised = False
        except SystemExit as exc:
            raised = True
            assert exc.code == 1

    assert raised
