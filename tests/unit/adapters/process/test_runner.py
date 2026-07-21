"""Targeted tests for the process execution adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from partio.adapters.process import runner


def test_resolve_executable_uses_existing_path(monkeypatch, tmp_path: Path) -> None:
    """Path-like executable names should be validated via filesystem checks."""
    tool = tmp_path / "tool.exe"
    tool.write_text("", encoding="utf-8")

    monkeypatch.setattr(runner.os, "access", lambda path, _mode: Path(path) == tool)
    resolved = runner.resolve_executable(str(tool))

    assert Path(resolved) == tool.resolve()


def test_resolve_executable_uses_which_and_missing_cases(monkeypatch) -> None:
    """Non-path executable names should resolve via which and fail closed when missing."""
    monkeypatch.setattr(
        runner.shutil,
        "which",
        lambda name: "C:/bin/demo.exe" if name == "demo" else None,
    )

    assert runner.resolve_executable("demo") == "C:/bin/demo.exe"

    with pytest.raises(FileNotFoundError):
        runner.resolve_executable("missing")


def test_run_resolved_builds_command_and_handles_empty(monkeypatch) -> None:
    """run_resolved should resolve first token and forward args to subprocess.run."""
    calls: list[tuple[list[str], dict]] = []

    class DummyCompleted:
        returncode = 0

    monkeypatch.setattr(runner, "resolve_executable", lambda name: f"C:/resolved/{name}.exe")

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return DummyCompleted()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    result = runner.run_resolved(["python", "-V"], capture_output=True, text=True)

    assert result.returncode == 0
    assert calls[0][0] == ["C:/resolved/python.exe", "-V"]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True

    with pytest.raises(ValueError, match="empty command"):
        runner.run_resolved([])
