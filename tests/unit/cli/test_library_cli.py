"""Tests for the library CLI command group (add, list, remove)."""

from __future__ import annotations

import pytest

from part_io.cli.commands.library import _store
from part_io.cli.commands.library import add as library_add
from part_io.cli.commands.library import list as library_list
from part_io.cli.commands.library import remove as library_remove
from part_io.cli.output import ExitCode
from part_io.core.ports import AudioPathKind


@pytest.fixture(autouse=True)
def _library_path(tmp_path, monkeypatch):
    """Point the default store at a scratch file for every test in this module."""
    path = tmp_path / "library.json"
    monkeypatch.setattr(_store, "DEFAULT_LIBRARY_PATH", path)
    return path


def test_add_remembers_an_entry(capsys, tmp_path):
    """add() should write a new entry and print a confirmation."""
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")

    library_add.add(ctx=None, path=audio, label="Episode 1", kind=AudioPathKind.SOURCE)

    output = capsys.readouterr().out
    assert "Remembered Episode 1 (source)" in output
    assert len(_store.default_store().list_items()) == 1


def test_add_fails_when_path_missing(tmp_path):
    """add() should exit with USER_ERROR when the audio file doesn't exist."""
    missing = tmp_path / "missing.mp3"
    with pytest.raises(SystemExit) as exc_info:
        library_add.add(ctx=None, path=missing, label=None, kind=AudioPathKind.SOURCE)
    assert exc_info.value.code == ExitCode.USER_ERROR


def test_list_reports_no_entries_when_empty(capsys):
    """list_entries() should report no results and exit NO_RESULT when empty."""
    with pytest.raises(SystemExit) as exc_info:
        library_list.list_entries(ctx=None)
    assert exc_info.value.code == ExitCode.NO_RESULT
    assert "No library entries found." in capsys.readouterr().out


def test_list_prints_added_entries(capsys, tmp_path):
    """list_entries() should print every remembered entry."""
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    library_add.add(ctx=None, path=audio, label="Episode 1", kind=AudioPathKind.SAMPLE)
    capsys.readouterr()

    library_list.list_entries(ctx=None)

    output = capsys.readouterr().out
    assert "Episode 1" in output
    assert "sample" in output


def test_remove_deletes_an_entry(capsys, tmp_path):
    """remove() should delete the entry and print a confirmation."""
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    library_add.add(ctx=None, path=audio, label="Episode 1", kind=AudioPathKind.SOURCE)
    entry_id = _store.default_store().list_items()[0].id
    capsys.readouterr()

    library_remove.remove(ctx=None, item_id=entry_id)

    assert f"Removed {entry_id}" in capsys.readouterr().out
    assert _store.default_store().list_items() == []


def test_remove_fails_for_unknown_id():
    """remove() should exit with USER_ERROR for an id that was never added."""
    with pytest.raises(SystemExit) as exc_info:
        library_remove.remove(ctx=None, item_id="does-not-exist")
    assert exc_info.value.code == ExitCode.USER_ERROR
