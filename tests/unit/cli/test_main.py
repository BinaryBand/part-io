"""Tests for the cli layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer as typer_mod
from typer.testing import CliRunner

from partio.cli.main import _show_picker, app
from partio.cli.registry import get_commands
from partio.cli.select import GO_BACK

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
    """The registry should list exactly the audio, feed, and library commands."""
    commands = get_commands()
    names = {(entry.group, entry.name) for entry in commands}
    assert names == {
        ("audio", "search"),
        ("audio", "locate"),
        ("audio", "review"),
        ("audio", "bootstrap"),
        ("feed", "add"),
        ("feed", "download"),
        ("feed", "list"),
        ("feed", "remove"),
        ("library", "add"),
        ("library", "list"),
        ("library", "remove"),
    }


def test_registry_entries_have_help() -> None:
    """Every registered command must carry a non-empty help string."""
    for entry in get_commands():
        assert entry.help, f"{entry.name} is missing a help string"


def test_registry_entries_have_group() -> None:
    """Every registered command must have a group."""
    for entry in get_commands():
        assert entry.group, f"{entry.name} is missing a group"


# -- picker navigation -----------------------------------------------------


def test_esc_during_arg_walkthrough_redisplays_the_menu() -> None:
    """GO_BACK from the walkthrough loops back to the command picker."""
    with (
        patch(
            "partio.cli.main.select_one", side_effect=["audio bootstrap", "library list"]
        ) as pick,
        patch("partio.cli.prompting.prompt_for_args", side_effect=[GO_BACK, []]) as walk,
        patch("partio.cli.main.app") as app_mock,
        patch("partio.cli.main.Console.print"),
    ):
        _show_picker()

    # Menu shown twice, walkthrough run twice, command invoked once.
    assert pick.call_count == 2
    assert walk.call_count == 2
    app_mock.assert_called_once()
    assert app_mock.call_args.args[0][:2] == ["library", "list"]


def test_esc_at_the_menu_quits() -> None:
    """The top-level menu has nothing to go back to, so esc exits."""
    with (
        patch("partio.cli.main.select_one", return_value=GO_BACK),
        patch("partio.cli.main.Console.print"),
        pytest.raises(typer_mod.Exit),
    ):
        _show_picker()


def test_cancelling_the_walkthrough_quits() -> None:
    """ctrl-c during the walkthrough exits rather than looping forever."""
    with (
        patch("partio.cli.main.select_one", return_value="audio bootstrap"),
        patch("partio.cli.prompting.prompt_for_args", return_value=None),
        patch("partio.cli.main.Console.print"),
        pytest.raises(typer_mod.Exit),
    ):
        _show_picker()
