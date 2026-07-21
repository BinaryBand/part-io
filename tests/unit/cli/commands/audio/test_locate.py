"""Tests for the cli.commands.audio.locate module."""

from __future__ import annotations

import pytest

from partio.adapters.audio.matcher import BestMatch
from partio.cli.commands.audio import locate as audio_locate


def test_audio_locate_main_prints_best_match(monkeypatch, capsys, tmp_path):
    """The locate CLI should print the best match with prominence."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_locate,
        "find_best_sample_match",
        lambda **_kwargs: BestMatch(
            start_seconds=3.0, end_seconds=6.0, duration_seconds=3.0, score=0.95, prominence=3.2
        ),
    )

    audio_locate.locate(source=source, sample=sample, ctx=None)

    output = capsys.readouterr().out
    assert "3.000s -> 6.000s" in output
    assert "score=0.9500" in output
    assert "prominence=3.20" in output


def test_audio_locate_main_rejects_low_prominence(monkeypatch, capsys, tmp_path):
    """Low prominence should exit with code 1."""
    source = tmp_path / "source.mp3"
    sample = tmp_path / "sample.mp3"
    source.write_bytes(b"source")
    sample.write_bytes(b"sample")

    monkeypatch.setattr(
        audio_locate,
        "find_best_sample_match",
        lambda **_kwargs: BestMatch(
            start_seconds=3.0, end_seconds=6.0, duration_seconds=3.0, score=0.5, prominence=0.1
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        audio_locate.locate(source=source, sample=sample, min_prominence=2.0, ctx=None)

    assert excinfo.value.code == 1
    assert "No confident match found" in capsys.readouterr().out
