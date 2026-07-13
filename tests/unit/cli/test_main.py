"""Tests for the cli layer."""

from __future__ import annotations

from typer.testing import CliRunner

from part_io.cli.main import app

runner = CliRunner()


def test_hello_greets_by_name() -> None:
    result = runner.invoke(app, ["World"])
    assert result.exit_code == 0
    assert "Hello, World" in result.stdout
