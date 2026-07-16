"""Tests for the utils.exec module."""

from __future__ import annotations

from partio.utils.exec import resolve_executable, run_resolved


def test_resolve_executable_imports() -> None:
    """Verify resolve_executable is importable."""
    assert resolve_executable is not None


def test_run_resolved_imports() -> None:
    """Verify run_resolved is importable."""
    assert run_resolved is not None
