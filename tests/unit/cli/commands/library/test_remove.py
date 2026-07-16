"""Tests for the cli.commands.library.remove module."""

from __future__ import annotations

from part_io.cli.commands.library.remove import remove


def test_library_remove_command_imports() -> None:
    """Verify the library remove command is importable."""
    assert remove is not None
