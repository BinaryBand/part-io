"""Tests for the cli.commands.library.add module."""

from __future__ import annotations

from partio.cli.commands.library.add import add


def test_library_add_command_imports() -> None:
    """Verify the library add command is importable."""
    assert add is not None
