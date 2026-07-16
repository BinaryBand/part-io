"""Tests for the utils.coverage module."""

from __future__ import annotations

from partio.utils.coverage import cleanup_coverage_temp_files


def test_cleanup_coverage_temp_files_imports() -> None:
    """Verify cleanup_coverage_temp_files is importable."""
    assert cleanup_coverage_temp_files is not None
