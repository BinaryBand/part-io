"""Tests for the cli.commands.library.list module."""

from __future__ import annotations

from partio.cli.commands.library.list import list_entries


def test_library_list_command_imports() -> None:
    """Verify the library list command is importable."""
    assert list_entries is not None
