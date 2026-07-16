"""Tests for the cli.output module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from partio.cli.output import (
    ExitCode,
    bundle_summary,
    emit,
    fail,
    locate_result,
    match_line,
    no_match,
    seed_written,
)

# -- ExitCode --------------------------------------------------------------


def test_exit_code_values() -> None:
    """ExitCode members have the expected numeric values."""
    assert ExitCode.OK == 0
    assert ExitCode.NO_RESULT == 1
    assert ExitCode.USER_ERROR == 2
    assert ExitCode.INTERNAL == 70


# -- emit ------------------------------------------------------------------


def test_emit_string(capsys) -> None:
    """emit() with a plain string prints it."""
    emit("hello world")
    assert capsys.readouterr().out.strip() == "hello world"


def test_emit_list(capsys) -> None:
    """emit() with a list prints each element on its own line."""
    emit(["line1", "line2", "line3"])
    lines = capsys.readouterr().out.strip().split("\n")
    assert lines == ["line1", "line2", "line3"]


def test_emit_json_string(capsys) -> None:
    """emit() with as_json=True wraps a string in a JSON object."""
    emit("hello", as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data == {"message": "hello"}


def test_emit_json_dict(capsys) -> None:
    """emit() with as_json=True and a dict serializes it."""
    emit({"key": "value", "count": 42}, as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data == {"key": "value", "count": 42}


def test_emit_json_list(capsys) -> None:
    """emit() with as_json=True and a list serializes it."""
    emit(["a", "b"], as_json=True)
    data = json.loads(capsys.readouterr().out)
    assert data == ["a", "b"]


# -- fail ------------------------------------------------------------------


def test_fail_exits_with_user_error() -> None:
    """fail() raises SystemExit with USER_ERROR code."""
    with pytest.raises(SystemExit) as excinfo:
        fail(RuntimeError("boom"))
    assert excinfo.value.code == ExitCode.USER_ERROR


def test_fail_prints_to_stderr(capsys) -> None:
    """fail() prints the exception message to stderr."""
    with pytest.raises(SystemExit):
        fail(RuntimeError("something went wrong"))
    assert "something went wrong" in capsys.readouterr().err


# -- existing formatters ---------------------------------------------------


def test_no_match() -> None:
    """Standard 'no results' message with a simple label."""
    assert no_match("matches") == "No matches found."


def test_no_match_custom_label() -> None:
    """The label is interpolated as-is."""
    assert no_match("jingles") == "No jingles found."


def test_match_line() -> None:
    """Match formatting includes start, end, and score."""
    result = match_line(1.234, 5.678, 0.9123)
    assert result == "1.234s -> 5.678s (score=0.9123)"


def test_match_line_zero_score() -> None:
    """Zero score is formatted correctly."""
    result = match_line(0.0, 1.0, 0.0)
    assert result == "0.000s -> 1.000s (score=0.0000)"


def test_locate_result() -> None:
    """Locate output includes prominence."""
    result = locate_result(1.234, 5.678, 0.9123, 3.45)
    assert result == "1.234s -> 5.678s (score=0.9123, prominence=3.45)"


def test_locate_result_zero_prominence() -> None:
    """Zero prominence is formatted correctly."""
    result = locate_result(0.0, 1.0, 0.5, 0.0)
    assert result == "0.000s -> 1.000s (score=0.5000, prominence=0.00)"


def test_bundle_summary() -> None:
    """Bundle summary returns exactly four lines."""
    lines = bundle_summary(
        bundle_dir=Path("downloads/review/bundle1"),
        selected_count=10,
        total_matches=25,
        manifest_path=Path("downloads/review/bundle1/matches_manifest.csv"),
        labels_path=Path("downloads/review/bundle1/match_labels.json"),
    )
    assert len(lines) == 4
    assert lines[0] == "Bundle: downloads/review/bundle1"
    assert lines[1] == "Exported clips: 10 (from 25 total matches)"
    assert "Manifest:" in lines[2]
    assert "Labels:" in lines[3]


def test_bundle_summary_zero_selected() -> None:
    """Bundle summary handles zero selected clips."""
    lines = bundle_summary(
        bundle_dir=Path("out"),
        selected_count=0,
        total_matches=0,
        manifest_path=Path("out/m.csv"),
        labels_path=Path("out/l.json"),
    )
    assert lines[1] == "Exported clips: 0 (from 0 total matches)"


def test_seed_written() -> None:
    """Seed confirmation includes onset, offset, and path."""
    result = seed_written(Path("static/jingles/ep1_seed.mp3"), 42.5, 45.2)
    assert result == "jingle 42.500s -> 45.200s written to static/jingles/ep1_seed.mp3"


def test_seed_written_integer_times() -> None:
    """Integer seconds are padded to three decimal places."""
    result = seed_written(Path("out.mp3"), 1, 2)
    assert result == "jingle 1.000s -> 2.000s written to out.mp3"
