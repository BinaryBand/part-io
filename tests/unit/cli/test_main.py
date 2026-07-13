"""Tests for the cli layer."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from part_io.cli.main import app
from part_io.cli.registry import get_commands

runner = CliRunner()

_COMMAND_NAME_RE = re.compile(r"^[a-z]+(-[a-z]+)*$")


def test_app_shows_help_with_flag() -> None:
    """--help should succeed and list the registered subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "search-audio" in result.stdout
    assert "locate-audio" in result.stdout
    assert "review-audio" in result.stdout
    assert "bootstrap-audio" in result.stdout


def test_app_version_flag() -> None:
    """--version should print the package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "part-io" in result.stdout


def test_search_audio_subcommand_help() -> None:
    """Each subcommand should have --help available."""
    result = runner.invoke(app, ["search-audio", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_locate_audio_subcommand_help() -> None:
    result = runner.invoke(app, ["locate-audio", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_review_audio_subcommand_help() -> None:
    result = runner.invoke(app, ["review-audio", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


def test_bootstrap_audio_subcommand_help() -> None:
    result = runner.invoke(app, ["bootstrap-audio", "--help"])
    assert result.exit_code == 0
    assert "source" in result.stdout.lower()


# -- registry tests --------------------------------------------------------


def test_registry_contains_all_commands() -> None:
    """The registry should list exactly the four audio commands."""
    commands = get_commands()
    names = {entry.name for entry in commands}
    assert names == {"search-audio", "locate-audio", "review-audio", "bootstrap-audio"}


def test_registry_entries_have_help() -> None:
    """Every registered command must carry a non-empty help string."""
    for entry in get_commands():
        assert entry.help, f"{entry.name} is missing a help string"


# -- naming convention tests -----------------------------------------------


def test_command_names_follow_verb_noun_convention() -> None:
    """All command names must be lowercase kebab-case: ``<verb>-<noun>``."""
    for entry in get_commands():
        assert _COMMAND_NAME_RE.match(entry.name), (
            f"Command name {entry.name!r} does not match "
            f"the expected pattern <verb>-<noun> (lowercase-kebab-case)"
        )
        parts = entry.name.split("-")
        assert len(parts) >= 2, (
            f"Command name {entry.name!r} should have at least a verb and a noun "
            f"(e.g. 'search-audio')"
        )
