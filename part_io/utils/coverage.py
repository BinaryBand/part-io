"""Helpers for safely cleaning stale coverage temp artifacts."""

from __future__ import annotations

from pathlib import Path


def cleanup_coverage_temp_files(root: Path | None = None) -> int:
    """Delete top-level ``.coverage.*`` files from *root* and return count.

    This intentionally avoids recursive traversal to prevent runaway cleanup
    behavior in large workspaces.
    """
    workspace_root = root or Path.cwd()
    removed = 0

    for candidate in workspace_root.iterdir():
        if not candidate.name.startswith(".coverage."):
            continue

        # Only remove files/symlinks; never recurse into directories.
        if not (candidate.is_file() or candidate.is_symlink()):
            continue

        try:
            candidate.unlink()
            removed += 1
        except OSError:
            continue

    return removed


__all__ = ["cleanup_coverage_temp_files"]
