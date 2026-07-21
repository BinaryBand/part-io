"""Tests for the cli.commands.audio.review module."""

from __future__ import annotations

import json

from partio.adapters.audio.matcher import AudioMatch
from partio.cli.commands.audio import review as audio_review


def test_audio_review_main_writes_bundle(monkeypatch, capsys, tmp_path):
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

    audio_review.review(
        ctx=None,
        source=source,
        sample=sample,
        output_root=tmp_path / "review",
        bundle_name="bundle",
        max_clips=1,
    )

    bundle_dir = tmp_path / "review" / "bundle"
    manifest_path = bundle_dir / "matches_manifest.csv"
    labels_path = bundle_dir / "match_labels.json"
    output = capsys.readouterr().out

    assert bundle_dir.exists()
    assert manifest_path.exists()
    assert labels_path.exists()
    assert "Exported clips: 1 (from 2 total matches)" in output


def test_audio_review_main_writes_interactive_labels(monkeypatch, tmp_path):
    """With --interactive, the review CLI should write a completed labels file."""
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
            AudioMatch(start_seconds=12.0, end_seconds=15.0, duration_seconds=3.0, score=0.7),
        ],
    )
    monkeypatch.setattr(audio_review, "_extract_clip", lambda **_kwargs: None)
    monkeypatch.setattr(
        "partio.cli.commands.audio._auditor.play_audio_segment", lambda **_kwargs: None
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    audio_review.review(
        ctx=None,
        source=source,
        sample=sample,
        output_root=tmp_path / "review",
        bundle_name="interactive",
        interactive=True,
    )

    labels_path = tmp_path / "review" / "interactive" / "match_labels.json"
    assert labels_path.exists()


def test_audio_review_main_default_writes_empty_template(monkeypatch, tmp_path):
    """Without --interactive, the review CLI writes an empty labels template."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_review,
        "find_audio_sample_matches",
        lambda **_kwargs: [
            AudioMatch(start_seconds=2.0, end_seconds=5.0, duration_seconds=3.0, score=0.9),
        ],
    )
    monkeypatch.setattr(audio_review, "_extract_clip", lambda **_kwargs: None)

    audio_review.review(
        ctx=None,
        source=source,
        sample=sample,
        output_root=tmp_path / "review",
        bundle_name="default",
    )

    labels_path = tmp_path / "review" / "default" / "match_labels.json"
    data = json.loads(labels_path.read_text())
    assert data["true_positive_indices"] == []
    assert data["false_positive_indices"] == []
