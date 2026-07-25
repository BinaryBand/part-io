"""Tests for the cli.commands.audio.cut module."""

from __future__ import annotations

from pathlib import Path

import pytest

from partio.adapters.audio.matcher import AudioMatch, BestMatch
from partio.cli.commands.audio import cut as audio_cut
from partio.core.audio_cut import KeepSegment


def _make_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "episode.mp3"
    opening = tmp_path / "open.mp3"
    closing = tmp_path / "close.mp3"
    for path in (source, opening, closing):
        path.write_bytes(b"audio")
    return source, opening, closing


def _patch_matches(monkeypatch, *, opening: BestMatch | None, closing: BestMatch | None) -> None:
    def _locate(*, source_path, sample_path, step_seconds):  # noqa: ARG001
        return opening if sample_path.name == "open.mp3" else closing

    monkeypatch.setattr(audio_cut, "find_best_sample_match", _locate)


def test_audio_cut_removes_span_and_reports(monkeypatch, capsys, tmp_path) -> None:
    """A successful cut writes the surviving audio and prints a summary."""
    source, opening, closing = _make_sources(tmp_path)
    _patch_matches(
        monkeypatch,
        opening=BestMatch(100.0, 105.0, 5.0, 0.95, 4.0),
        closing=BestMatch(160.0, 166.0, 6.0, 0.93, 3.5),
    )
    monkeypatch.setattr(audio_cut, "audio_duration_seconds", lambda _p: 600.0)

    captured: dict[str, object] = {}

    def _fake_cut(*, source_path, destination_path, keep_segments) -> None:
        captured["source_path"] = source_path
        captured["destination_path"] = destination_path
        captured["keep_segments"] = keep_segments

    monkeypatch.setattr(audio_cut, "cut_segments", _fake_cut)

    out_path = tmp_path / "edited.mp3"
    audio_cut.cut(source=source, opening=opening, closing=closing, output=out_path, ctx=None)

    assert captured["keep_segments"] == [
        KeepSegment(0.0, 100.0),
        KeepSegment(166.0, 600.0),
    ]
    assert captured["destination_path"] == out_path
    output = capsys.readouterr().out
    assert "Removed 66.000s (100.000s -> 166.000s)" in output
    assert "Kept 534.000s of 600.000s" in output
    assert str(out_path) in output


def test_audio_cut_defaults_output_beside_source(monkeypatch, capsys, tmp_path) -> None:
    """With no --output, the cut is written next to the source as <stem>_cut.mp3."""
    source, opening, closing = _make_sources(tmp_path)
    _patch_matches(
        monkeypatch,
        opening=BestMatch(10.0, 12.0, 2.0, 0.9, 4.0),
        closing=BestMatch(30.0, 33.0, 3.0, 0.9, 4.0),
    )
    monkeypatch.setattr(audio_cut, "audio_duration_seconds", lambda _p: 100.0)

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        audio_cut,
        "cut_segments",
        lambda **kwargs: captured.update(kwargs),
    )

    audio_cut.cut(source=source, opening=opening, closing=closing, ctx=None)

    assert captured["destination_path"] == tmp_path / "episode_cut.mp3"
    assert str(tmp_path / "episode_cut.mp3") in capsys.readouterr().out


def test_audio_cut_no_confident_opening_match(monkeypatch, capsys, tmp_path) -> None:
    """A missing/weak opening jingle exits with NO_RESULT and does not cut."""
    source, opening, closing = _make_sources(tmp_path)
    _patch_matches(
        monkeypatch,
        opening=BestMatch(10.0, 12.0, 2.0, 0.5, 0.1),
        closing=BestMatch(30.0, 33.0, 3.0, 0.9, 4.0),
    )

    def _unexpected_cut(**_kwargs) -> None:
        raise AssertionError("cut_segments should not be called when no jingle matches")

    monkeypatch.setattr(audio_cut, "cut_segments", _unexpected_cut)

    with pytest.raises(SystemExit) as excinfo:
        audio_cut.cut(source=source, opening=opening, closing=closing, min_prominence=2.0, ctx=None)

    assert excinfo.value.code == 1
    assert "No confident match for the opening jingle" in capsys.readouterr().out


def test_audio_cut_refuses_existing_output(monkeypatch, capsys, tmp_path) -> None:
    """An existing output without --overwrite is a user error."""
    source, opening, closing = _make_sources(tmp_path)
    _patch_matches(
        monkeypatch,
        opening=BestMatch(10.0, 12.0, 2.0, 0.9, 4.0),
        closing=BestMatch(30.0, 33.0, 3.0, 0.9, 4.0),
    )
    monkeypatch.setattr(audio_cut, "audio_duration_seconds", lambda _p: 100.0)
    monkeypatch.setattr(
        audio_cut,
        "cut_segments",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not cut")),
    )

    out_path = tmp_path / "edited.mp3"
    out_path.write_bytes(b"existing")

    with pytest.raises(SystemExit) as excinfo:
        audio_cut.cut(source=source, opening=opening, closing=closing, output=out_path, ctx=None)

    assert excinfo.value.code == 2
    assert "already exists" in capsys.readouterr().err


def test_audio_cut_default_output_helper() -> None:
    """The default output path appends _cut before the suffix."""
    assert audio_cut._default_output_path(Path("/media/ep.mp3")) == Path("/media/ep_cut.mp3")


def _patch_all_matches(
    monkeypatch, *, openings: list[AudioMatch], closings: list[AudioMatch]
) -> None:
    def _find(*, source_path, sample_path, score_threshold, step_seconds, dedupe_overlap):  # noqa: ARG001
        return openings if sample_path.name == "open.mp3" else closings

    monkeypatch.setattr(audio_cut, "find_audio_sample_matches", _find)


def test_audio_cut_all_removes_every_break(monkeypatch, capsys, tmp_path) -> None:
    """--all pairs each opening with the next closing and removes every break."""
    source, opening, closing = _make_sources(tmp_path)
    _patch_all_matches(
        monkeypatch,
        openings=[AudioMatch(100.0, 105.0, 5.0, 0.9), AudioMatch(300.0, 305.0, 5.0, 0.9)],
        closings=[AudioMatch(160.0, 166.0, 6.0, 0.9), AudioMatch(360.0, 366.0, 6.0, 0.9)],
    )
    monkeypatch.setattr(audio_cut, "audio_duration_seconds", lambda _p: 600.0)

    captured: dict[str, object] = {}
    monkeypatch.setattr(audio_cut, "cut_segments", lambda **kwargs: captured.update(kwargs))

    out_path = tmp_path / "edited.mp3"
    audio_cut.cut(
        source=source, opening=opening, closing=closing, output=out_path, all_breaks=True, ctx=None
    )

    assert captured["keep_segments"] == [
        KeepSegment(0.0, 100.0),
        KeepSegment(166.0, 300.0),
        KeepSegment(366.0, 600.0),
    ]
    output = capsys.readouterr().out
    assert "Removed 2 break(s), 132.000s total:" in output
    assert "1. 100.000s -> 166.000s" in output
    assert "2. 300.000s -> 366.000s" in output


def test_audio_cut_all_no_breaks(monkeypatch, capsys, tmp_path) -> None:
    """--all with no pairable jingles exits NO_RESULT and does not cut."""
    source, opening, closing = _make_sources(tmp_path)
    _patch_all_matches(
        monkeypatch,
        openings=[],
        closings=[AudioMatch(160.0, 166.0, 6.0, 0.9)],
    )
    monkeypatch.setattr(audio_cut, "audio_duration_seconds", lambda _p: 600.0)
    monkeypatch.setattr(
        audio_cut,
        "cut_segments",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not cut")),
    )

    with pytest.raises(SystemExit) as excinfo:
        audio_cut.cut(source=source, opening=opening, closing=closing, all_breaks=True, ctx=None)

    assert excinfo.value.code == 1
    assert "No ad breaks found" in capsys.readouterr().out
