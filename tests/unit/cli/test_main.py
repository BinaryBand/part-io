"""Tests for the cli layer."""

from __future__ import annotations

from typer.testing import CliRunner

from part_io.cli.main import app

runner = CliRunner()


def test_app_shows_help_with_flag() -> None:
    """--help should succeed and list the registered subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audio-search" in result.stdout
    assert "audio-locate" in result.stdout
    assert "audio-review" in result.stdout
    assert "audio-bootstrap" in result.stdout


def test_app_version_flag() -> None:
    """--version should print the package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "part-io" in result.stdout


def test_audio_search_subcommand_help() -> None:
    """Each subcommand should have --help available."""
    result = runner.invoke(app, ["audio-search", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_audio_locate_subcommand_help() -> None:
    result = runner.invoke(app, ["audio-locate", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_audio_review_subcommand_help() -> None:
    result = runner.invoke(app, ["audio-review", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_audio_bootstrap_subcommand_help() -> None:
    result = runner.invoke(app, ["audio-bootstrap", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()
