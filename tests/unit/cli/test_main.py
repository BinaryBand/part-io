"""Tests for the cli layer."""

from __future__ import annotations

from typer.testing import CliRunner

from partio.cli.main import app
from partio.cli.registry import get_commands

runner = CliRunner()


def test_app_shows_help_with_flag() -> None:
    """--help should succeed and list the audio group."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audio" in result.stdout


def test_app_version_flag() -> None:
    """--version should print the package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "partio" in result.stdout


def test_audio_group_shows_help() -> None:
    """The audio sub-app should list its commands."""
    result = runner.invoke(app, ["audio", "--help"])
    assert result.exit_code == 0
    assert "search" in result.stdout
    assert "locate" in result.stdout
    assert "review" in result.stdout
    assert "bootstrap" in result.stdout


def test_audio_search_subcommand_help() -> None:
    """Each subcommand should have --help available."""
    result = runner.invoke(app, ["audio", "search", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_audio_locate_subcommand_help() -> None:
    result = runner.invoke(app, ["audio", "locate", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_audio_review_subcommand_help() -> None:
    result = runner.invoke(app, ["audio", "review", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_audio_bootstrap_subcommand_help() -> None:
    result = runner.invoke(app, ["audio", "bootstrap", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


# -- registry tests --------------------------------------------------------


def test_registry_contains_all_commands() -> None:
    """The registry should list exactly the four audio commands and the library commands."""
    commands = get_commands()
    names = {(entry.group, entry.name) for entry in commands}
    assert names == {
        ("audio", "search"),
        ("audio", "locate"),
        ("audio", "review"),
        ("audio", "bootstrap"),
        ("library", "add"),
        ("library", "list"),
        ("library", "remove"),
        ("library", "download"),
    }


def test_registry_entries_have_help() -> None:
    """Every registered command must carry a non-empty help string."""
    for entry in get_commands():
        assert entry.help, f"{entry.name} is missing a help string"


def test_registry_entries_have_group() -> None:
    """Every registered command must have a group."""
    for entry in get_commands():
        assert entry.group, f"{entry.name} is missing a group"
