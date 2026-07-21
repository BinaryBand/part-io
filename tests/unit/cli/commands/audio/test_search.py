"""Tests for the cli.commands.audio.search module."""

from __future__ import annotations

from partio.adapters.audio.matcher import AudioMatch
from partio.cli.commands.audio import search as audio_search


def test_audio_search_main_prints_matches(monkeypatch, capsys, tmp_path):
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

    audio_search.search(source=source, sample=sample, ctx=None)

    output = capsys.readouterr().out
    assert "1.230s -> 4.560s" in output
    assert "score=0.9100" in output
