"""Targeted tests for process helpers and coverage artifact cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from part_io.utils import coverage as cov_utils
from part_io.utils import exec as exec_utils


def test_cleanup_coverage_temp_files_only_removes_top_level_files(tmp_path: Path) -> None:
    """Cleanup helper should remove only matching top-level temp artifacts."""
    keep_file = tmp_path / ".coverage"
    keep_file.write_text("base", encoding="utf-8")

    remove_one = tmp_path / ".coverage.a"
    remove_one.write_text("x", encoding="utf-8")

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_nested = nested_dir / ".coverage.b"
    nested_nested.write_text("x", encoding="utf-8")

    pseudo_dir = tmp_path / ".coverage.dir"
    pseudo_dir.mkdir()

    removed = cov_utils.cleanup_coverage_temp_files(tmp_path)

    assert removed == 1
    assert keep_file.exists()
    assert not remove_one.exists()
    assert nested_nested.exists()
    assert pseudo_dir.exists()


def test_resolve_executable_uses_existing_path(monkeypatch, tmp_path: Path) -> None:
    """Path-like executable names should be validated via filesystem checks."""
    tool = tmp_path / "tool.exe"
    tool.write_text("", encoding="utf-8")

    monkeypatch.setattr(exec_utils.os, "access", lambda path, mode: Path(path) == tool)
    resolved = exec_utils.resolve_executable(str(tool))

    assert Path(resolved) == tool.absolute()


def test_resolve_executable_preserves_symlink_path(monkeypatch, tmp_path: Path) -> None:
    """Symlink paths should not be dereferenced to keep venv interpreter semantics."""
    target = tmp_path / "python-real"
    target.write_text("", encoding="utf-8")
    link = tmp_path / "python"
    link.symlink_to(target)

    monkeypatch.setattr(exec_utils.os, "access", lambda path, mode: Path(path) == link)
    resolved = exec_utils.resolve_executable(str(link))

    assert Path(resolved) == link.absolute()


def test_resolve_executable_uses_which_and_missing_cases(monkeypatch) -> None:
    """Non-path executable names should resolve via which and fail closed when missing."""
    monkeypatch.setattr(
        exec_utils.shutil,
        "which",
        lambda name: "C:/bin/demo.exe" if name == "demo" else None,
    )

    assert exec_utils.resolve_executable("demo") == "C:/bin/demo.exe"

    with pytest.raises(FileNotFoundError):
        exec_utils.resolve_executable("missing")


def test_run_resolved_builds_command_and_handles_empty(monkeypatch) -> None:
    """run_resolved should resolve first token and forward args to subprocess.run."""
    calls: list[tuple[list[str], dict]] = []

    class DummyCompleted:
        returncode = 0

    monkeypatch.setattr(exec_utils, "resolve_executable", lambda name: f"C:/resolved/{name}.exe")

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return DummyCompleted()

    monkeypatch.setattr(exec_utils.subprocess, "run", fake_run)

    result = exec_utils.run_resolved(["python", "-V"], capture_output=True, text=True)

    assert result.returncode == 0
    assert calls[0][0] == ["C:/resolved/python.exe", "-V"]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True

    with pytest.raises(ValueError, match="empty command"):
        exec_utils.run_resolved([])


def test_launch_resolved_builds_popen_command(monkeypatch) -> None:
    """launch_resolved should resolve first token and call Popen with full path."""
    launched: list[list[str]] = []

    class DummyPopen:
        def __init__(self, cmd, **kwargs):
            launched.append(cmd)

    monkeypatch.setattr(exec_utils, "resolve_executable", lambda name: f"/resolved/{name}")
    monkeypatch.setattr(exec_utils.subprocess, "Popen", DummyPopen)

    result = exec_utils.launch_resolved(["ffplay", "-nodisp", "file.mp3"])

    assert launched == [["/resolved/ffplay", "-nodisp", "file.mp3"]]
    assert isinstance(result, DummyPopen)


def test_launch_resolved_raises_on_empty_command(monkeypatch) -> None:
    """launch_resolved should raise ValueError for an empty command."""
    with pytest.raises(ValueError, match="empty command"):
        exec_utils.launch_resolved([])
