"""Tests for the cli package init (handle_cli_error)."""

from __future__ import annotations

import pytest

from part_io.cli import handle_cli_error


def test_handle_cli_error_exits_with_code_2() -> None:
    """handle_cli_error prints the exception and exits with code 2."""
    with pytest.raises(SystemExit, match="2"):
        handle_cli_error(RuntimeError("boom"))
